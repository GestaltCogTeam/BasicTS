# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parent  # PROJECT_DIR
BASICTS_DIR = PROJECT_DIR / "basicts"  # BASICTS_DIR
sys.path.append(PROJECT_DIR.as_posix())
sys.path.append(BASICTS_DIR.as_posix())
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightning.pytorch.cli import LightningCLI

from basicts.data.tsf_datamodule import TimeSeriesForecastingModule
from basicts.model import BasicTimeSeriesForecastingModule
# from .baselines


class BasicTSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)

        # parser.link_arguments("model.init_args.null_val", "data.regular_settings[NULL_VAL]")
        # parser.link_arguments("model.init_args.history_len", "data.init_args.input_len")
        # parser.link_arguments("data.init_args.prediction_len", "data.init_args.output_len")


def run():
    cli = BasicTSCLI(
        run=True,
        trainer_defaults={},
        parser_kwargs={"parser_mode": "omegaconf"},  # pip install omegaconf
        save_config_kwargs={"overwrite": True, "save_to_log_dir": True},
    )
    if cli.subcommand in ("fit", "validate") and not cli.trainer.fast_dev_run:
        # 被动执行了 fit 或者 validate，追加一个 test
        cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    run()
