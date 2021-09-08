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

if __name__ == "__main__":
    from gym_multistory_fourrooms import FourRooms
    from gym_multistory_fourrooms.utils.fr_utils import grid_to_rgb
    from time import sleep
    test = FourRooms(16, grid_z=2, obs_n=3)
    o = test.reset()
    for t in range(100):
        o, r, d, _ = test.step(test.action_space.sample())
        test.render(mode='human', idx=np.arange(1))
        sleep(0.2)

    o = test.reset()
    o2 = test._get_obs()[np.arange(16)]
    print(o2)
    test = grid_to_rgb(o2, 16)
    import pygame
    pygame.init()
    size = test.shape[:-1]
    screen = pygame.display.set_mode(size)
    sfc = pygame.surfarray.make_surface(test)
    screen.blit(sfc, (0,0))
    # pygame.display.update()
    sleep(0.5)
    print(test.shape)
    print(3)