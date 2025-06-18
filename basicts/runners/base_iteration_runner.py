import inspect
import logging
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import setproctitle
import torch
import torch.distributed
from easytorch.config import get_ckpt_save_dir
from easytorch.core.checkpoint import (backup_last_ckpt, clear_ckpt, load_ckpt,
                                       save_ckpt)
from easytorch.core.data_loader import build_data_loader, build_data_loader_ddp
from easytorch.core.meter_pool import MeterPool
from easytorch.device import to_device
from easytorch.utils import (TimePredictor, get_local_rank, get_logger,
                             is_master, master_only, set_env)
from packaging import version
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import InfiniteGenerator, get_dataset_name
from ..utils.misc import \
    convert_iteration_save_strategy_to_epoch_save_strategy as \
    convert_save_strategy
from .optim import build_lr_scheduler, build_optim


class BaseIterationRunner(metaclass=ABCMeta):
    """Base Runner for EasyTorch.
    A runner that uses iteration as the fundamental training unit.
    
    In conventional deep learning, the training cycle is typically based on epochs.
    However, some models use iteration-based training cycles.
    For example, in the training of large models,
        the vast amount of data often makes it impractical to use epochs as the fundamental unit.
    Instead, iterations are used, with the Dataloader continuously drawing data from the dataset.
    Training stops once the maximum number of iterations is reached.
    
    Other Features:
        - Support torch.compile
        - support early stopping
        - Infinite Train Dataloader

    Note:
        - Move the backward and learning rate scheduler into the train_iter function.
        - Save model checkpoints every `self.val_interval`.
    """

    def __init__(self, cfg: Dict) -> None:
        # default logger
        self.logger = get_logger('easytorch')

        # set env
        set_env(cfg.get('ENV', {}))

        # param
        self.model_name = cfg['MODEL.NAME']
        self.ckpt_save_dir = get_ckpt_save_dir(cfg)
        self.logger.info('Set ckpt save dir: \'{}\''.format(self.ckpt_save_dir))
        self.ckpt_save_strategy = None
        self.num_iterations = None
        self.start_iteration = None

        self.val_interval = cfg.get('VAL.INTERVAL', None)
        self.test_interval = cfg.get('TEST.INTERVAL', None)
        self.to_running_device = to_device

        # create checkpoint save dir
        if not os.path.isdir(self.ckpt_save_dir):
            os.makedirs(self.ckpt_save_dir)

        # create model
        self.model = self.build_model(cfg)

        # declare optimizer and lr_scheduler
        self.optim = None
        self.scheduler = None

        # declare data loader
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.save_results = cfg.get('EVAL', {}).get('SAVE_RESULTS', False)

        # declare meter pool
        self.meter_pool = None

        # declare tensorboard_writer
        self.tensorboard_writer = None

        # support early stopping
        # NOTE: If the project has been stopped early and its configuration is rerun,
        #           training will resume from the last saved checkpoint.
        #       This feature is designed primarily for the convenience of users,
        #           allowing them to continue training seamlessly after an interruption.
        self.early_stopping_patience = cfg.get('TRAIN', {}).get('EARLY_STOPPING_PATIENCE', None)
        self.current_patience = self.early_stopping_patience
        assert self.early_stopping_patience is None or self.early_stopping_patience > 0, 'Early stopping patience must be a positive integer.'

        # set process title
        proctitle_name = f"{cfg['MODEL'].get('NAME')}({get_dataset_name(cfg)})"
        setproctitle.setproctitle(f'{proctitle_name}@BasicTS')

    def define_model(self, cfg: Dict) -> nn.Module:
        """
        Define the model architecture based on the configuration.

        Args:
            cfg (Dict): Configuration dictionary containing model settings.

        Returns:
            nn.Module: The model architecture.
        """

        return cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM'])

    def build_model(self, cfg: Dict) -> nn.Module:
        """Build model.

        Initialize model by calling ```self.define_model```,
        Moves model to the GPU.

        If DDP is initialized, initialize the DDP wrapper.

        Args:
            cfg (Dict): config

        Returns:
            model (nn.Module)
        """

        self.logger.info('Building model.')
        model = self.define_model(cfg)
        model = self.to_running_device(model)

        # complie model
        if cfg.get('TRAIN.COMPILE_MODEL', False):
            # get current torch version
            current_version = torch.__version__
            # torch.compile() is only available in torch>=2.0
            if version.parse(current_version) >= version.parse('2.0'):
                self.logger.info('Compile model with torch.compile')
                model = torch.compile(model)
            else:
                self.logger.warning(f'torch.compile requires PyTorch 2.0 or higher. Current version: {current_version}. Skipping compilation.')

        # DDP
        if torch.distributed.is_initialized():
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                find_unused_parameters=cfg.get('MODEL.DDP_FIND_UNUSED_PARAMETERS', False)
            )
        return model

    def train(self, cfg: Dict) -> None:
        """Train model.

        Train process:
        [init_training]
        for in train_iteration
            [on_iteration_start]
            [train_iters]
            [on_iteration_end] ------> Iteration Val: val every n iteration
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
        [on_training_end]

        Args:
            cfg (Dict): config
        """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_iteration, self.num_iterations)

        iteration = 0
        # training loop
        for iteration in tqdm(range(self.start_iteration+1, self.num_iterations+1),\
                                    initial=self.start_iteration+1, total=self.num_iterations, mininterval=0):
            # early stopping check
            if self.check_early_stopping():
                break

            self.on_iteration_start(iteration)
            iteration_start_time = time.time()
            # start training
            self.model.train()

            # feed the train_iters with data_loader to perform gradient accumulation
            self.train_iters(iteration=iteration, dataloader=self.train_data_loader)

            iteration_end_time = time.time()
            # iteration time
            self.update_iteration_meter('train/iter_time', iteration_end_time - iteration_start_time)
            self.on_iteration_end(iteration)

            expected_end_time = train_time_predictor.get_expected_end_time(iteration)

            # estimate training finish time
            if iteration < self.num_iterations and iteration % self.val_interval == 0:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end(cfg=cfg, train_iteration=iteration + 1)

    def init_training(self, cfg: Dict) -> None:
        """Initialize training

        Args:
            cfg (Dict): config
        """

        self.logger.info('Initializing training.')

        # init training param
        self.num_iterations = cfg['TRAIN.NUM_ITERATIONS']
        self.start_iteration = 0
        self.ckpt_save_strategy = convert_save_strategy(cfg.get('TRAIN.CKPT_SAVE_STRATEGY'), self.val_interval)
        self.best_metrics = {}
        self.clip_grad_param = cfg.get('TRAIN.CLIP_GRAD_PARAM')
        if self.clip_grad_param is not None:
            self.logger.info('Set clip grad, param: {}'.format(self.clip_grad_param))

        # train data loader
        self.train_data_loader = InfiniteGenerator(self.build_train_data_loader(cfg))
        self.register_iteration_meter('train/iter_time', 'train', '{:.2f} (s)', plt=False)

        # create optim
        self.optim = self.build_optim(cfg['TRAIN.OPTIM'], self.model)
        self.logger.info('Set optim: {}'.format(self.optim))

        # create lr_scheduler
        self.init_lr_scheduler(cfg)

        # fine tune
        if cfg.has('TRAIN.FINETUNE_FROM'):
            self.load_model(cfg['TRAIN.FINETUNE_FROM'], cfg.get('TRAIN.FINETUNE_STRICT_LOAD', True))
            self.logger.info('Start fine tuning')

        # resume
        self.load_model_resume()

        # init tensorboard(after resume)
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_iteration + 1) if self.start_iteration != 0 else None
            )

        # init validation
        if cfg.has('VAL'):
            self.init_validation(cfg)

        # init test
        if cfg.has('TEST'):
            self.init_test(cfg)

    def build_optim(self, optim_cfg: Dict, model: nn.Module) -> optim.Optimizer:
        """Build optimizer from `optim_cfg`.

        Args:
            optim_cfg (Dict): optimizer config
            model (nn.Module): model

        Returns:
            optim.Optimizer: optimizer
        """

        if optim_cfg['TYPE'] != 'AdamW_Fused_nanoGPT':
            return build_optim(optim_cfg, model)
        else:
            # https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L263
            optim_param = optim_cfg['PARAM']
            learning_rate = optim_param['lr']
            weight_decay = optim_param.get('weight_decay', 0.1)
            betas = optim_param.get('betas', (0.9, 0.95))
            device_type = 'cuda'    # only supported on GPU

            # start with all of the candidate parameters
            param_dict = dict(self.model.named_parameters())
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f'num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
            print(f'num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = {'fused': True} if use_fused else {}
            # the learning rate will be updated in the training process
            # so we do not need to consider resuming from a checkpoint
            return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    def build_train_data_loader(self, cfg: Dict) -> DataLoader:
        """Build train dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader``` or
        ```build_data_loader_ddp``` when DDP is initialized

        Args:
            cfg (Dict): config

        Returns:
            train data loader (DataLoader)
        """

        self.logger.info('Building training data loader.')
        dataset = self.build_train_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN.DATA'])
        else:
            return build_data_loader(dataset, cfg['TRAIN.DATA'])

    @staticmethod
    @abstractmethod
    def build_train_dataset(cfg: Dict) -> Dataset:
        """It must be implement to build dataset for training.

        Args:
            cfg (Dict): config

        Returns:
            train dataset (Dataset)
        """

        pass

    def build_inference_dataset(self, cfg: Dict, input_data: Union[str, list], context_length: int, prediction_length: int) -> Dataset:
        """
        Build the inference dataset.

        Args:
            cfg (Dict): Configuration dictionary.
            input_data (Union[str, list]): The input data file path or data list for inference.
            context_length (int): The length of the context for inference.
            prediction_length (int): The length of the prediction for inference.

        Returns:
            Dataset: The inference dataset.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """

        raise NotImplementedError('build_inference_dataset method must be implemented.')

    def init_lr_scheduler(self, cfg: Dict) -> None:
        """
        Initialize lr_scheduler

        Args:
            cfg (Dict): config
        """
        # create lr_scheduler
        if cfg.has('TRAIN.LR_SCHEDULER'):
            self.scheduler = build_lr_scheduler(cfg['TRAIN.LR_SCHEDULER'], self.optim)
            self.logger.info('Set lr_scheduler: {}'.format(self.scheduler))
            self.register_iteration_meter('train/lr', 'train', '{:.2e}')

    @master_only
    def init_inference(self, cfg: Dict, input_data: Union[str, list], context_length: int, prediction_length: int) -> None:
        """
        Initialize the inference data loader and related settings.

        Args:
            cfg (Dict): Configuration dictionary.
            input_data (Union[str, list]): The input data file path or data list for inference.
            context_length (int): The length of the context for inference.
            prediction_length (int): The length of the prediction for inference.
        """

        self.inference_dataset = self.build_inference_dataset(cfg, input_data, context_length, prediction_length)
        self.inference_dataset_loader = DataLoader(self.inference_dataset, batch_size=1, shuffle=False)
        self.register_iteration_meter('inference/time', 'inference', '{:.2f} (s)', plt=False)

    def load_model(self, ckpt_path: str = None, strict: bool = True) -> None:
        """Load model state dict.
        if param `ckpt_path` is None, load the last checkpoint in `self.ckpt_save_dir`,
        else load checkpoint from `ckpt_path`

        Args:
            ckpt_path (str, optional): checkpoint path, default is None
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError) as e:
            raise OSError('Ckpt file does not exist') from e

    def load_model_resume(self, strict: bool = True) -> None:
        """Load last checkpoint in checkpoint save dir to resume training.

        Load model state dict.
        Load optimizer state dict.
        Load start iteration and set it to lr_scheduler.

        Args:
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_iteration = checkpoint_dict['iteration']
            if checkpoint_dict.get('best_metrics') is not None:
                self.best_metrics = checkpoint_dict['best_metrics']
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])

            if hasattr(self, 'amp_scaler'):
                self.amp_scaler.load_state_dict(checkpoint_dict['amp_scaler_state_dict'])

            self.logger.info('Resume training')
        except (IndexError, OSError, KeyError):
            pass

    @master_only
    def init_validation(self, cfg: Dict) -> None:
        """Initialize validation

        Args:
            cfg (Dict): config
        """

        self.logger.info('Initializing validation.')
        self.val_data_loader = self.build_val_data_loader(cfg)
        self.register_iteration_meter('val/time', 'val', '{:.2f} (s)', plt=False)

    def build_val_data_loader(self, cfg: Dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (Dict): config

        Returns:
            val data loader (DataLoader)
        """

        self.logger.info('Building val data loader.')
        dataset = self.build_val_dataset(cfg)
        return build_data_loader(dataset, cfg['VAL.DATA'])

    @staticmethod
    def build_val_dataset(cfg: Dict) -> Dataset:
        """It can be implement to build dataset for validation (not necessary).

        Args:
            cfg (Dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    @master_only
    def init_test(self, cfg: Dict) -> None:
        """
        Initialize the test data loader and related settings.

        Args:
            cfg (Dict): Configuration dictionary.
        """

        self.logger.info('Initializing test.')
        assert self.test_interval is None or (self.test_interval > 0), \
            f'Invalid test interval: {self.test_interval}'
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_iteration_meter('test/time', 'test', '{:.2f} (s)', plt=False)

    def build_test_data_loader(self, cfg: Dict) -> DataLoader:
        """
        Build the test data loader.

        Args:
            cfg (Dict): Configuration dictionary.

        Returns:
            DataLoader: The test data loader.
        """

        self.logger.info('Building test data loader.')
        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST.DATA'])

    def build_test_dataset(self, cfg: Dict) -> Dataset:
        """
        Build the test dataset.

        Args:
            cfg (Dict): Configuration dictionary.

        Returns:
            Dataset: The test dataset.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """

        raise NotImplementedError('build_test_dataset method must be implemented.')

    def on_iteration_start(self, iteration: int):
        """Callback at the start of an iteration.

        Args:
            iteration (int): current iteration
        """

        if iteration % self.val_interval == 0:
            # print iteration num
            self.logger.info('Iteration {:d} / {:d}'.format(iteration, self.num_iterations))
            # update lr meter
        if self.scheduler is not None:
            self.update_iteration_meter('train/lr', self.scheduler.get_last_lr()[0])

    @abstractmethod
    def train_iters(self, iteration: int, dataloader: DataLoader) -> torch.Tensor:
        """It must be implement to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            iteration (int): current iteration.
            dataloader (torch.utils.data.DataLoader):dataloader.

        Returns:
            loss (torch.Tensor)
        """

        pass

    def on_iteration_end(self, iteration: int) -> None:
        """Callback at the end of an iteration.

        Args:
            iteration (int): current iteration.
        """

        # tensorboard plt meters
        self.plt_iteration_meters('train', iteration, value_type='last')
        if iteration % self.val_interval == 0:
            # print train meters
            self.print_iteration_meters('train')
            # validate
            if self.val_data_loader is not None:
                self.validate(train_iteration=iteration)
                # save model
                self.save_model(iteration)
            # reset meters
            self.reset_iteration_meters()

        if self.test_interval is not None and iteration % self.test_interval == 0 and self.test_data_loader is not None:
            self.test_pipeline(train_iteration=iteration)

    def on_training_end(self, cfg: Dict, train_iteration: Optional[int] = None) -> None:
        """Callback at the end of the training process.
        
        Args:
            cfg (Dict): Configuration.
            train_iteration (Optional[int]): End iteration if in training process.
        """

        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()

        if hasattr(cfg, 'TEST'):
            # evaluate the best model on the test set
            best_model_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_val_{}.pt'.format(self.model_name, self.target_metrics.replace('/', '_'))
            )
            self.logger.info('Evaluating the best model on the test set.')
            self.load_model(ckpt_path=best_model_path, strict=True)
            self.test_pipeline(cfg=cfg, train_iteration=train_iteration, save_metrics=True, save_results=self.save_results)

    def get_ckpt_path(self, iteration: int) -> str:
        """Get checkpoint path.

        The format is "{ckpt_save_dir}/{model_name}_{iteration}"

        Args:
            iteration (int): current iteration.

        Returns:
            checkpoint path (str)
        """

        iteration_str = str(iteration).zfill(len(str(self.num_iterations)))
        ckpt_name = '{}_{}.pt'.format(self.model_name, iteration_str)
        return os.path.join(self.ckpt_save_dir, ckpt_name)

    @master_only
    def save_model(self, iteration: int) -> None:
        """Save checkpoint every iteration.

        checkpoint format is {
            'iteration': current iteration ([1, num_iterations]),
            'model_state_dict': state_dict of model,
            'optim_state_dict': state_dict of optimizer
        }

        Decide whether to delete the last checkpoint by the checkpoint save strategy.

        Args:
            iteration (int): current iteration.
        """

        model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_dict = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'best_metrics': self.best_metrics
        }

        # save learning rate scheduler if available
        if self.scheduler is not None:
            ckpt_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        if hasattr(self, 'amp_scaler'):
            ckpt_dict['amp_scaler_state_dict'] = self.amp_scaler.state_dict()

        # backup last iteration
        eqv_epoch = iteration // self.val_interval
        eqv_save_strategy = self.ckpt_save_strategy
        last_ckpt_path = self.get_ckpt_path(iteration - self.val_interval)
        backup_last_ckpt(last_ckpt_path, eqv_epoch, eqv_save_strategy)

        # save ckpt
        ckpt_path = self.get_ckpt_path(iteration)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # clear ckpt every 10*self.val_interval iterations or in the end
        if iteration % (10 * self.val_interval) == 0 or iteration == self.num_iterations:
            clear_ckpt(self.ckpt_save_dir)

    @torch.no_grad()
    def validate(self, cfg: Dict = None, train_iteration: Optional[int] = None) -> None:
        """Validate model.

        Args:
            cfg (Dict, optional): config
            train_iteration (int, optional): current iteration if in training process.
        """

        # init validation if not in training process
        if train_iteration is None:
            self.init_validation(cfg)

        self.logger.info('Start validation.')

        self.on_validating_start(train_iteration)

        val_start_time = time.time()
        self.model.eval()

        # tqdm process bar
        data_iter = tqdm(self.val_data_loader, leave=False)
        # data_iter = self.val_data_loader

        # val loop
        for iter_index, data in enumerate(data_iter):
            self.val_iters(iter_index, data)

        val_end_time = time.time()
        self.update_iteration_meter('val/time', val_end_time - val_start_time)
        # print val meters
        self.print_iteration_meters('val')
        if train_iteration is not None:
            # tensorboard plt meters
            self.plt_iteration_meters('val', train_iteration // self.val_interval)

        self.on_validating_end(train_iteration)

    @master_only
    def on_validating_start(self, train_iteration: Optional[int]) -> None:
        """Callback at the start of validating.

        Args:
            train_iteration (Optional[int]): current iteration if in training process.
        """

        pass

    @master_only
    def on_validating_end(self, train_iteration: Optional[int]) -> None:
        """Callback at the end of validating.

        Args:
            train_iteration (Optional[int]): current iteration if in training process.
        """

        pass

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]) -> None:
        """It can be implement to define validating detail (not necessary).

        Args:
            iter_index (int): current iter.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
        """

        raise NotImplementedError()

    def test_pipeline(self, cfg: Optional[Dict] = None, train_iteration: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> None:
        """Test model.

        Args:
            cfg (Dict, optional): Configuration dictionary. Defaults to None.
            train_iteration (int, optional): Current iteration during training. Defaults to None.
            save_metrics (bool, optional): Save the test metrics. Defaults to False.
            save_results (bool, optional): Save the test results. Defaults to False.
        """

        if train_iteration is None and cfg is not None:
            self.init_test(cfg)

        self.logger.info('Start test.')
        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # execute the test process
        self.test(train_iteration=train_iteration, save_metrics=save_metrics, save_results=save_results)

        test_end_time = time.time()
        self.update_iteration_meter('test/time', test_end_time - test_start_time)

        self.print_iteration_meters('test')
        if train_iteration is not None:
            self.plt_iteration_meters('test', train_iteration // self.test_interval)

        # logging here for intuitiveness
        if save_results:
            self.logger.info(f'Test results saved to {os.path.join(self.ckpt_save_dir, "test_results.npz")}.')
        if save_metrics:
            self.logger.info(f'Test metrics saved to {os.path.join(self.ckpt_save_dir, "test_metrics.json")}.')

        self.on_test_end()

    @torch.no_grad()
    @master_only
    def test(self, train_iteration: Optional[int] = None, save_metrics: bool = False, save_results: bool = False) -> Dict:
        """
        Define the details of the testing process.

        Args:
            train_iteration (int, optional): Current epoch during training. Defaults to None.
            save_metrics (bool, optional): Save the test metrics. Defaults to False.
            save_results (bool, optional): Save the test results. Defaults to False.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """

        raise NotImplementedError('test method must be implemented.')

    @torch.no_grad()
    @master_only
    def inference_pipeline(self, cfg: Optional[Dict] = None, input_data: Union[str, list] = '', output_data_file_path: str = '', context_length=0, prediction_length=0) -> tuple:
        """
        The complete inference process.

        Args:
            cfg (Dict, optional): Configuration dictionary.
            input_data (Union[str, list], optional): The input data file path or data list for inference.
            output_data_file_path (str, optional): The output data file path. Defaults to '' meaning no output file.
            context_length (int, optional): Context length for inference, only used for utfs models. Defaults to 0.
            prediction_length (int, optional): Prediction length for inference, only used for utfs models. Defaults to 0.
        """

        # if isinstance(input_data, str):
        #     pass

        self.init_inference(cfg, input_data, context_length, prediction_length)

        self.on_inference_start()

        inference_start_time = time.time()
        self.model.eval()

        # execute the inference process
        result = self.inference(save_result_path=output_data_file_path)

        inference_end_time = time.time()
        self.update_iteration_meter('inference/time', inference_end_time - inference_start_time)

        self.print_iteration_meters('inference')

        # logging here for intuitiveness
        if output_data_file_path:
            self.logger.info(f'inference results saved to {output_data_file_path}.')

        self.on_inference_end()

        return result

    def inference(self, save_result_path: str = '') -> tuple:
        """
        Define the details of the inference process.

        Args:
            save_result_path (str, optional): The output data file path. Defaults to '' meaning no output file.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """

        raise NotImplementedError('test method must be implemented.')

    @master_only
    def on_test_start(self) -> None:
        """Callback at the start of testing."""

        pass

    @master_only
    def on_inference_start(self) -> None:
        """Callback at the start of inference."""

        pass

    @master_only
    def on_test_end(self) -> None:
        """Callback at the end of testing."""

        pass

    @master_only
    def on_inference_end(self) -> None:
        """Callback at the end of inference."""

        pass

    @master_only
    def save_best_model(self, iteration: int, metric_name: str, greater_best: bool = True) -> None:
        """Save the best model while training.

        Examples:
            >>> def on_validating_end(self, train_iteration: Optional[int]):
            >>>     if train_iteration is not None:
            >>>         self.save_best_model(train_iteration, 'val/loss', greater_best=False)

        Args:
            iteration (int): current iteration.
            metric_name (str): metric name used to measure the model, must be registered in `iteration_meter`.
            greater_best (bool, optional): `True` means greater value is best, such as `acc`
                `False` means lower value is best, such as `loss`. Defaults to True.
        """

        metric = self.meter_pool.get_avg(metric_name)
        best_metric = self.best_metrics.get(metric_name)
        if best_metric is None or (metric > best_metric if greater_best else metric < best_metric):
            self.best_metrics[metric_name] = metric
            model = self.model.module if isinstance(self.model, DDP) else self.model
            ckpt_dict = {
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': self.optim.state_dict(),
                'best_metrics': self.best_metrics
            }
            # save learning rate scheduler if available
            if self.scheduler is not None:
                ckpt_dict['scheduler_state_dict'] = self.scheduler.state_dict()

            if hasattr(self, 'amp_scaler'):
                ckpt_dict['amp_scaler_state_dict'] = self.amp_scaler.state_dict()

            ckpt_path = os.path.join(
                self.ckpt_save_dir,
                '{}_best_{}.pt'.format(self.model_name, metric_name.replace('/', '_'))
            )
            save_ckpt(ckpt_dict, ckpt_path, self.logger)
            self.current_patience = self.early_stopping_patience # reset patience
        else:
            if self.early_stopping_patience is not None:
                self.current_patience -= 1

    def init_logger(self, logger: logging.Logger = None, logger_name: str = None,
                    log_file_name: str = None, log_level: int = logging.INFO) -> None:
        """Initialize logger.

        Args:
            logger (logging.Logger, optional): specified logger.
            logger_name (str, optional): specified name of logger.
            log_file_name (str, optional): logger file name.
            log_level (int, optional): log level, default is INFO.
        """

        if logger is not None:
            self.logger = logger
        elif logger_name is not None:
            if log_file_name is not None:
                log_file_name = '{}_{}.log'.format(log_file_name, time.strftime('%Y%m%d%H%M%S', time.localtime()))
                log_file_path = os.path.join(self.ckpt_save_dir, log_file_name)
            else:
                log_file_path = None
            self.logger = get_logger(logger_name, log_file_path, log_level)
        else:
            raise TypeError('At least one of logger and logger_name is not None')

    @master_only
    def register_iteration_meter(self, name, meter_type, fmt='{:f}', plt=True) -> None:
        if self.meter_pool is None:
            self.meter_pool = MeterPool()
        self.meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_iteration_meter(self, name, value, n=1) -> None:
        self.meter_pool.update(name, value, n)

    @master_only
    def print_iteration_meters(self, meter_type) -> None:
        self.meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def plt_iteration_meters(self, meter_type, step, value_type = 'avg') -> None:
        self.meter_pool.plt_meters(meter_type, step, self.tensorboard_writer, value_type)

    @master_only
    def reset_iteration_meters(self) -> None:
        self.meter_pool.reset()

    def check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.early_stopping_patience is not None and self.current_patience <= 0:
            self.logger.info('Early stopping.')
            return True
        return False
