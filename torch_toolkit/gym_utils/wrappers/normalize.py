__all__ = ['VectorObservationNormalizer']

import gym.spaces
from gym import Wrapper
from functools import partial
from gym.vector import VectorEnvWrapper
import numpy as np
import torch as th
from ...utils.normalization import RunningMeanStd, RunningMeanStdTorch

class VectorObservationNormalizer(VectorEnvWrapper):
    """Class to normalize vectorized environment observations"""
    def __init__(self, venv, clip_obs=10.):
        super().__init__(venv)
        assert isinstance(self.single_observation_space, gym.spaces.Box)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        is_th = hasattr(self, 'device') and self.device is not None
        cls = partial(RunningMeanStdTorch, device=self.device) if is_th else RunningMeanStd
        self.ob_rms = cls(shape=self.single_observation_space.shape)
        self.clip_fn = partial(th.clamp, min=-clip_obs, max=clip_obs) if is_th else partial(np.clip, a_min=-clip_obs, a_max=clip_obs)
        self.epsilon = 1e-8

    def step(self, action):
        o, r, d, info = self.env.step(action)
        return self._obfilt(o), r, d, info

    def reset(self):
        return self._obfilt(self.env.reset())

    def _obfilt(self, obs):
        self.ob_rms.update(obs)
        obs = self.clip_fn((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon))
        return obs


