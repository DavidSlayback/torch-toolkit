__all__ = ['TimeLimit']
import gym
import numpy as np


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = np.zeros(self.num_envs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self.is_vector_env:
            done[self._elapsed_steps >= self._max_episode_steps] = True
            self._elapsed_steps[done] = 0  # Auto-reset
        else: done = self._elapsed_steps >= self._max_episode_steps
        return observation, reward, done, info

    def reset(self, **kwargs):
        if self.is_vector_env:
            self._elapsed_steps[:] = 0
        else:
            self._elapsed_steps = 0
        return self.env.reset(**kwargs)