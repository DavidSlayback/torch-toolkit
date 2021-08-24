import gym
from gym.vector import SyncVectorEnv
from functools import partial
from torch_toolkit.gym_utils import VectorObservationNormalizer, PyTorchVectorWrapper, decorrelate_env

if __name__ == "__main__":
    e = partial(gym.make, 'CartPole-v0')
    s = SyncVectorEnv([e for _ in range(8)])
    p = PyTorchVectorWrapper(s)
    pn = VectorObservationNormalizer(p)
    ret = decorrelate_env(p, 250, 1)
    o1 = pn.reset()