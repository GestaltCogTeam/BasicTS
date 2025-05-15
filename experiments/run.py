# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[1]  # PROJECT_DIR: BasicTS
BASICTS_DIR = PROJECT_DIR / "basicts"  # BASICTS_DIR: BasicTS/basicts
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(BASICTS_DIR.as_posix())

from lightning.pytorch.cli import LightningCLI

# from basicts.data.tsf_datamodule import TimeSeriesForecastingModule
# from basicts.model import BasicTimeSeriesForecastingModule

class BasicTSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)

        parser.link_arguments("data.init_args.dataset_name", "model.init_args.dataset_name")


def run():
    cli = BasicTSCLI(
        run=True,
        trainer_defaults={},
        
        parser_kwargs={"parser_mode": "omegaconf",
            "default_config_files": [(BASICTS_DIR/"configs"/"default.yaml").as_posix()],
        },
        save_config_kwargs={"overwrite": True, "save_to_log_dir": True},
    )
    logger = cli.trainer.logger

    # Log hyperparameters
    trainer_hparam_names = ['max_epochs', 'min_epochs', 'precision', 'overfit_batches', 'gradient_clip_val', 'gradient_clip_algorithm', 'accelerator', 'strategy', 'limit_train_batches', 'limit_val_batches', 'limit_test_batches']
    trainer_hparams = {k: cli.config_dump['trainer'][k] for k in trainer_hparam_names}
    logger.log_hyperparams(cli.datamodule.hparams)
    logger.log_hyperparams(cli.model.hparams)
    logger.log_hyperparams(trainer_hparams)


    if cli.subcommand in ("fit", "validate") and not cli.trainer.fast_dev_run:
        # 被动执行了 fit 或者 validate，追加一个 test
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    run()
