from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class BasicTSTaskFlow(ABC):
    """BasicTS Task Flow"""

    @abstractmethod
    def preprocess(self, runner: 'BasicTSRunner', data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        pass

    @abstractmethod
    def postprocess(self, runner: 'BasicTSRunner', forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """Run the task flow"""

        pass

    @abstractmethod
    def get_weight(self, forward_return: Dict[str, Any]) -> float:
        """Get the weight of the forward return"""

        pass
