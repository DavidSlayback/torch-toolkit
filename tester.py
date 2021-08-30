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
    T, B = 20, 30
    from torch_toolkit.buffer import AgnosticRolloutBuffer
    b = AgnosticRolloutBuffer(T)
    b_th = AgnosticRolloutBuffer(T, 'cpu')
    b_th_c = AgnosticRolloutBuffer(T, 'cuda')
    init_dict = {'next_o': np.random.rand(B, 3, 3), 'prev_a': np.random.randint(5, size=(B,)), 'lp': th.rand(B, 5), 'v': th.rand(B,)}
    add_dict = {'o': np.random.rand(B, 3, 3), 'a': np.random.randint(5, size=(B,)), 'lp': th.rand(B, 5), 'v': th.rand(B,) }
    b.add(**init_dict)
    b_th.add(**init_dict)
    b_th_c.add(**init_dict)
    for t in range(T):
        b.add(**add_dict)
        b_th.add(**add_dict)
        b_th_c.add(**add_dict)
    for mb in b.mb_sample(4):
        print(mb)
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