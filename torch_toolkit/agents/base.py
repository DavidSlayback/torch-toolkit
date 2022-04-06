
from typing import Union, NamedTuple
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
        ...

    @abc.abstractmethod
    def observe_first(self, observation: GType):
        """Get first observation from environment"""
        ...

    @abc.abstractmethod
    def observe(self,
                action: GType,
                next_observation: GType
                ):
        """Observe a timestep of data"""
        ...

    @abc.abstractmethod
    def update(self):
        """Update model"""
        ...

# Configs hold all the fixed constants
class A2CConfig(NamedTuple):
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gae: float = 0.95  # GAE lambda
    gamma: float = 0.99  # Discount factor
    reward_scale: float = 1.  # Multiple rewards by this
    normalize_obs: bool = False  # Normalize incoming observations before passing to network
    normalize_advantage: bool = True  # Normalize advantages after rollout
    max_grad_norm: float = 0.5  # Max gradient norm
    vf_coef: float = 0.5  # Value loss coefficient


class PPOConfig(A2CConfig):
    update_epochs: int = 4
    clip_coef: float = 0.2
    num_minibatches: int = 4
