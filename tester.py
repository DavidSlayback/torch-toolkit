from functools import partial

import torch
import torch.nn as nn

if __name__ == "__main__":
    from dm_control import suite
    import gym
    from gym.vector import SyncVectorEnv
    import torch.nn.functional as F
    from torch_toolkit.gym_utils.wrappers.dm_env_conversion import GymFromDMEnv
    from torch_toolkit.typing import TensorDict
    from torch_toolkit.networks.misc import FlattenDict
    from torch_toolkit.networks.intakes import build_intake_from_gym_env
    from torch_toolkit.networks.outputs import build_action_head_from_gym_env, BernoulliHead, batched_index
    e = SyncVectorEnv([partial(gym.make, 'Pendulum-v1') for _ in range(4)])
    e2 = SyncVectorEnv([partial(gym.make, 'CartPole-v1') for _ in range(4)])
    m, i = build_intake_from_gym_env(e)
    o = torch.tensor(e.reset())
    o2 = torch.tensor(e2.reset())
    m1 = torch.jit.script(build_action_head_from_gym_env(e, continuous_parameterization='gaussian_ind')(o.shape[-1]))
    m2 = torch.jit.script(build_action_head_from_gym_env(e2)(o2.shape[-1]))
    action, scaled_action, logits = m1.sample(o)
    lp = m1.log_prob(logits, action)
    ent = m1.entropy(logits)
    h2 = torch.jit.script(BernoulliHead(o.shape[-1], 4))
    idx = torch.arange(4)
    term, scaled_term, term_logits = h2.sample(o, idx)
    lp = h2.log_prob(term_logits, term)
    ent = h2.entropy(term_logits)
    # action, logits = m2.sample(o2)
    # lp = m2.log_prob(logits, action)
    # ent = m2.entropy(logits)
    print(3)