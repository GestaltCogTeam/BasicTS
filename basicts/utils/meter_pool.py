import logging
from typing import Any, Dict, Tuple, Union

from torch.utils.tensorboard import SummaryWriter

from ..metrics import METRIC_METER


class MeterPool:
    """Meter container
    """

    def __init__(self):
        self._pool: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, meter_type: str, fmt: str = '{:f}', plt: bool = True):
        """Add a meter to meter pool.
        Args:
            name (str): meter name.
            meter_type (str): meter type.
            fmt (str): meter output format.
            plt (bool): set ```True``` to plot it in tensorboard
                when calling ```plt_meters```.
        """

        if name in self._pool:
            raise ValueError(f'Meter {name} already existed.')

        # name: type/metric or type/metric@h{i}
        metric = name.split('/')[1].split('@')[0] # get the metric name
        handle_meter = 'default' if metric not in METRIC_METER else metric

        self._pool[name] = {
            'meter': METRIC_METER[handle_meter](),
            'index': len(self._pool.keys()),
            'format': fmt,
            'type': meter_type,
            'plt': plt
        }

    def update(self, name: str, value: Union[float, Tuple[float]] , n: int = 1):
        """Update average meter.

        Args:
            name (str): meter name.
            value (Union[float, Tuple[float]]): value.
            n: (int): num.
        """

        self._pool[name]['meter'].update(value, n)

    def get_value(self, name: str) -> float:
        """Get value.

        Args:
            name (str): meter name.

        Returns:
            avg (float)
        """

        return self._pool[name]['meter'].value

    def print_meters(self, meter_type: str, logger: logging.Logger = None):
        """Print the specified type of meters.

        Args:
            meter_type (str): meter type
            logger (logging.Logger): logger
        """

        print_list = []
        for i in range(len(self._pool.keys())):
            for name, value in self._pool.items():
                if value['index'] == i and value['type'] == meter_type:
                    print_list.append(
                        ('{}: ' + value['format']).format(name, value['meter'].value)
                    )
        print_str = 'Result <{}>: [{}]'.format(meter_type, ', '.join(print_list))
        if logger is None:
            print(print_str)
        else:
            logger.info(print_str)

    def plt_meters(self, meter_type: str, step: int, tensorboard_writer: SummaryWriter):
        """Plot the specified type of meters in tensorboard.

        Args:
            meter_type (str): meter type.
            step (int): Global step value to record
            tensorboard_writer (SummaryWriter): tensorboard SummaryWriter
        """

        for name, value in self._pool.items():
            if value['plt'] and value['type'] == meter_type:
                tensorboard_writer.add_scalar(name, value['meter'].value, global_step=step)
        tensorboard_writer.flush()

    def reset(self):
        """Reset all meters.
        """

        for _, value in self._pool.items():
            value['meter'].reset()
