from typing import NamedTuple, Union

import gym
import numpy as np
import torch.jit
from gym.vector import SyncVectorEnv
from functools import partial
from torch_toolkit.utils import to_th, th_stack
from torch_toolkit.gym_utils import VectorObservationNormalizer, PyTorchVectorWrapper, decorrelate_env, DMEnvFromGym, GymFromDMEnv
from torch_toolkit.networks.examples import BasePOMDPBody
from torch.nn.functional import mse_loss
import torch as th
import torch.nn as nn

if __name__ == "__main__":
    from dm_control import suite
    test = suite.load('quadruped', 'fetch')
    test_gym = GymFromDMEnv(test)
    test_dm = DMEnvFromGym(test_gym)
    print(3)