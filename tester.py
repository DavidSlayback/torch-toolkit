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
from torch.nn.functional import mse_loss
import torch as th
import torch.nn as nn
tjs = torch.jit.script

if __name__ == "__main__":
    from torch_toolkit.networks.rnn import update_state_with_mask
    from torch_toolkit.networks import ImageBOWEmbedding
    import gym_minigrid
    e = gym.make('MiniGrid-FourRooms-v0')
    o_n = e.observation_space
    tnet = ImageBOWEmbedding(o_n['image'].high.max(), 128)
    t = e.reset()['image']
    tx = th.transpose(th.transpose(th.from_numpy(t).unsqueeze(0), 1, 3), 2,3)
    tout = tnet(tx)
    from torch_toolkit.networks.examples import BasePOMDPBody
    tnet = BasePOMDPBody(18, 5, gru_layer_norm=True, gru_init_state_learnable=1)
    # module_init(tnet)
    n, p = list(zip(*tnet.named_modules()))
    t2 = nn.BatchNorm2d(768)
    mask = th.rand(30) > 0.5
    test = th.nn.GRUCell(64, 128)
    opt = th.optim.Adam(test.parameters(), 1e-3)
    target = torch.rand(30, 128)
    inp = torch.rand(30, 64)
    tout1 = test(inp); state = tout1.clone()
    state = update_state_with_mask(state, mask, test(inp[mask], state[mask]))

    tout2 = test(inp, state)
    loss = mse_loss(tout2, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
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