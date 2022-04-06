from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Env

import numpy as np
from .base import PPOConfig
from ..networks.intakes import build_intake_from_gym_env
from ..networks.outputs import build_separate_ff_actor_critic

class PPOAgent(nn.Module):
    def __init__(self,
                 envs: Env,
                 cfg: PPOConfig):
        super().__init__()







