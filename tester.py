import torch
import torch.nn as nn

if __name__ == "__main__":
    from dm_control import suite
    from torch_toolkit.gym_utils.wrappers.dm_env_conversion import GymFromDMEnv
    from torch_toolkit.typing import TensorDict
    from torch_toolkit.networks.misc import FlattenDict
    from torch_toolkit.networks.intakes import build_intake_from_gym_env
    test = suite.load('quadruped', 'fetch')
    e = GymFromDMEnv(test)
    espc = e.observation_space
    m, i = build_intake_from_gym_env(e)
    print(3)