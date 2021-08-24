__all__ = ['PyTorchWrapper', 'PyTorchVectorWrapper']

from typing import Union, Dict, Tuple
import gym
from gym import Env, Wrapper
from gym.vector import VectorEnvWrapper, VectorEnv
import torch as th
Tensor = th.Tensor

from ... import th_device, to_th, to_np


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