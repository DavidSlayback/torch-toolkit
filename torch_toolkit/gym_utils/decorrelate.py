__all__ = ['decorrelate_env']

from collections import deque
from functools import partial
from gym import Env
from ..utils import th_stack, np_stack


def decorrelate_env(env: Env, number_of_steps: int, number_of_steps_to_return: int = 1):
    """Decorrelate environments from each other by taking random steps

    Args:
        env: VectorEnv with single_action_space
        number_of_steps: Number of random actions to take
        number_of_steps_to_return: How many of the steps should be saved and returned for later use
    """
    assert hasattr(env, 'num_envs')
    buffer = deque(maxlen=number_of_steps_to_return)
    env.reset()
    stack_fn = partial(th_stack, device=env.device) if hasattr(env, 'device') else np_stack
    for t in range(number_of_steps):
        a = stack_fn(env.action_space.sample())
        o, r, d, info = env.step(a)
        buffer.append((o, a, r, d, info))  # Save obs, prev_a, prev_r, prev_d, info
    o, a, r, d, info = zip(*list(buffer))  # Convert to lists
    if number_of_steps_to_return == 1:
        return *(stack_fn(i).squeeze(0) for i in (o, a, r, d)), info
    else: return *(stack_fn(i) for i in (o, a, r, d)), info



