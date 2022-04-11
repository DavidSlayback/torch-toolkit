__all__ = ['PyTorchWrapper', 'PyTorchVectorWrapper', 'RecordEpisodeStatisticsTorch']

from collections import deque
from typing import Union, Dict, Tuple
import gym
from gym import Env, Wrapper
from gym.vector import VectorEnvWrapper, VectorEnv
import torch as th
Tensor = th.Tensor

from ...utils import th_device, to_th, to_np


class PyTorchWrapper(Wrapper):
    """Wrapper to convert gym environment i/o to torch tensors

    Args:
        env: gym.Env or gym.Wrapper instance of environment to wrap
        device: string or torch.Device (e.g., cpu, cuda, tpu) where tensors should be put
    """
    def __init__(self, env: Union[Wrapper, Env], device: Union[str, th.device] = 'cpu'):
        super().__init__(env)
        self.device = th_device(device)

    def reset(self) -> Tensor:
        return to_th(self.env.reset(), self.device)

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        o, r, d, info = self.env.step(to_np(actions))
        o, d = to_th((o,d))
        reward = to_th(r, self.device, th.float32)  # Override with float32
        return o, reward, d, info


class PyTorchVectorWrapper(VectorEnvWrapper):
    """Wrapper to convert gym vector environment i/o to torch tensors

    Args:
        env: VectorEnv or VectorEnvWrapper instance of environment to wrap
        device: string or torch.Device (e.g., cpu, cuda, tpu) where tensors should be put
    """
    def __init__(self, venv: Union[VectorEnv, VectorEnvWrapper], device: Union[str, th.device] = 'cpu'):
        super().__init__(venv)
        # Hack to get these set properly on the outside
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.device = th_device(device)

    def reset(self) -> Tensor:
        return to_th(self.env.reset(), self.device)

    def step_async(self, actions: Tensor):
        self.env.step_async(to_np(actions))

    def step_wait(self) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        o, r, d, info = self.env.step_wait()
        o, d = to_th((o,d))
        reward = to_th(r, self.device, th.float32)  # Override with float32
        return o, reward, d, info

import time

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    """Discounted return as well. Environment backed by tensors. Stat queues still in numpy for compatibility"""
    def __init__(self, env, discount: float, deque_size: int = 100):
        super(RecordEpisodeStatisticsTorch, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_brax_env = hasattr(self.env.unwrapped, '_state')
        self.is_vector_env = getattr(env, "is_vector_env", self.num_envs > 1)
        self.device = getattr(env, "device", 'cpu')
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = th.zeros(self.num_envs, dtype=th.float64, device=self.device)
        self.discounted_episode_returns = th.zeros(self.num_envs, dtype=th.float64, device=self.device)
        self.episode_lengths = th.zeros(self.num_envs, dtype=th.int32, device=self.device)
        self._cur_discount = th.ones(self.num_envs, dtype=th.float64, device=self.device)
        self.discount = discount or getattr(env, "discount", 0) or 1.
        # These are still numpy
        self.return_queue = deque(maxlen=deque_size)
        self.discounted_return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        self.discounted_episode_returns[:] = 0
        self._cur_discount[:] = 1
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(
            action
        )
        self.episode_returns += rewards
        self.discounted_episode_returns += rewards * self._cur_discount
        self.episode_lengths += 1
        self._cur_discount *= self.discount
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        d_idx = th.nonzero(dones).squeeze(-1)
        nd = d_idx.numel()
        if not self.is_brax_env: infos = list(infos)
        if nd:
            # Bring these parts over to cpu just once
            rets = to_np(self.episode_returns[d_idx])
            d_rets = to_np(self.discounted_episode_returns[d_idx])
            lens = to_np(self.episode_lengths[d_idx])
            # Brax environments: 1 info, dict of tensors
            if self.is_brax_env:
                infos['episode_info'] = {'r': self.episode_returns, 'l': self.episode_lengths,
                                         'dr': self.discounted_episode_returns,
                                         't': round(time.perf_counter() - self.t0, 6)}
            # Otherwise, list of infos
            else:
                for i, j in enumerate(d_idx):
                    infos[j] = infos[j].copy()
                    episode_info = {
                        "r": rets[i],
                        "l": lens[i],
                        "dr": d_rets[i],
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                    infos[j]["episode"] = episode_info
            self.return_queue.extend(rets)
            self.discounted_return_queue.extend(d_rets)
            self.length_queue.extend(lens)
            self.episode_count += nd
            self.episode_returns[d_idx] = 0
            self.episode_lengths[d_idx] = 0
            self.discounted_episode_returns[d_idx] = 0
            self._cur_discount[d_idx] = 1.
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )