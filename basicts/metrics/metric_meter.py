class AvgMeter:
    """Average meter.
    """

    def __init__(self):
        self._sum: float = 0.
        self._count: int = 0

    def reset(self):
        """Reset counter.
        """

        self._sum = 0.
        self._count = 0

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """

        self._sum += value * n
        self._count += n

    @property
    def value(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """

        return self._sum / self._count if self._count != 0 else 0


class RMSEMeter:
    """
    RMSE meter.
    This meter maintains **MSE** and calculate **RMSE** in the post process.
    """

    def __init__(self):
        self._mse: float = 0.
        self._count: int = 0

    def reset(self):
        """Reset counter.
        """

        self._mse = 0.
        self._count = 0

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """

        self._mse += value ** 2 * n
        self._count += n

    @property
    def value(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """

        mse = self._mse / self._count if self._count != 0 else 0

        return mse ** 0.5
