from typing import NamedTuple, Union

import dataclassy
import gym
import numpy as np
import torch.jit
from gym.vector import SyncVectorEnv
from functools import partial
from torch_toolkit.utils import to_th, th_stack
from torch_toolkit.gym_utils import VectorObservationNormalizer, PyTorchVectorWrapper, decorrelate_env
from torch_toolkit.networks.examples import BasePOMDPBody
import torch as th
tjs = torch.jit.script

if __name__ == "__main__":
    from torch_toolkit.utils import ArrayDataclassMixin, to_th, to_np
    @dataclassy.dataclass(slots=True)
    class Buffer(ArrayDataclassMixin):
        o: Union[th.Tensor, np.ndarray] = th.rand(20, 30, 5)
        d: Union[th.Tensor, np.ndarray] = th.rand(20, 30)
        r: Union[th.Tensor, np.ndarray] = np.random.rand(20, 30)

    test = Buffer()
    t = to_th(test, device='cuda')
    t = to_np(t)
    print(3)
    # from gym_pomdps import AutoresettingBatchPOMDP
    # e = AutoresettingBatchPOMDP(gym.make('POMDP-hallway-episodic-v0'), 256, time_limit=100)
    # from torch_toolkit.advantages import *
    # from torch_toolkit.gym_utils import *
    # from torch_toolkit.losses import *
    # from torch_toolkit.utils import *
    # from torch_toolkit.networks import *
    # agent = tjs(BasePOMDPBody(e.single_observation_space.n, e.single_action_space.n))
    # tout = agent(to_th(e.reset()), prev_action=th_stack(e.action_space.sample()))
    # x = th.rand(20, 64)
    # x2 = break_grad(x)
    # x2 = th.rand(30, 20, 64)
    # s = th.rand(20, 128)
    # d = th.randint(2, (30, 20)).float()