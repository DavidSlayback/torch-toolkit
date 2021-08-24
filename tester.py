import gym
import torch.jit
from gym.vector import SyncVectorEnv
from functools import partial
from torch_toolkit.gym_utils import VectorObservationNormalizer, PyTorchVectorWrapper, decorrelate_env
from torch_toolkit.networks import ResetGRU
import torch as th
tjs = torch.jit.script

if __name__ == "__main__":
    x = th.rand(20, 64)
    x2 = th.rand(30, 20, 64)
    s = th.rand(20, 128)
    d = th.randint(2, (30, 20)).float()
    net = tjs(ResetGRU(64, 128))
    net2 = tjs(ResetGRU(64, 128, layer_norm=True))
    net3 = tjs(ResetGRU(64, 128, 4))
    o1 = net(x, s)
    o2 = net(x2, s)
    o3 = net(x2)
    o4 = net(x2, s, d)
    print(3)
    # e = partial(gym.make, 'CartPole-v0')
    # s = SyncVectorEnv([e for _ in range(8)])
    # p = PyTorchVectorWrapper(s)
    # pn = VectorObservationNormalizer(p)
    # ret = decorrelate_env(p, 250, 1)
    # o1 = pn.reset()