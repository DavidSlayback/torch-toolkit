from typing import NamedTuple, Union

import gym
import numpy as np
import torch.jit
from gym.vector import SyncVectorEnv
from functools import partial
from torch_toolkit.utils import to_th, th_stack
from torch_toolkit.gym_utils import VectorObservationNormalizer, PyTorchVectorWrapper, decorrelate_env, DMEnvFromGym, GymFromDMEnv, DictObservabilityWrapper, VectorObservabilityWrapper, DictVectorObservabilityWrapper
from torch_toolkit.networks.examples import BasePOMDPBody
from torch.nn.functional import mse_loss
import torch as th
import torch.nn as nn

if __name__ == "__main__":
    from dm_control import suite
    import pybullet_envs
    test = suite.load('quadruped', 'fetch')
    test_gym = GymFromDMEnv(test)
    test_gym2 = DictObservabilityWrapper(test_gym, ['egocentric_state'])
    test_gym3 = DictVectorObservabilityWrapper(test_gym, {'egocentric_state': np.arange(20)})
    test_dm = DMEnvFromGym(test_gym)
    print(3)