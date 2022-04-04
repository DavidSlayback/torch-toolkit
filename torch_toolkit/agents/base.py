
from typing import Union
import abc


from ..typing import XArray, XArrayDict
GType = Union[XArray, XArrayDict]
class Agent:
    """Interface for an agent that can act.

    All agents responsible for maintaining parameters/models
    All agents responsible for maintaining buffers (rollout/replay)
    Recurrent agents are responsible for maintaining recurrent state
    """
    @abc.abstractmethod
    def select_action(self, observation: GType) -> GType:
        """Sample from policy, get action"""

    @abc.abstractmethod
    def observe_first(self, observation: GType):
        """Get first observation from environment"""

    @abc.abstractmethod
    def observe(self,
                action: GType,
                next_observation: GType
                ):
        """Observe a timestep of data"""

    @abc.abstractmethod
    def update(self):
        """Update model"""
