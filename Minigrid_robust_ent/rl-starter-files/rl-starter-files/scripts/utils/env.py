import gym
import gym_minigrid

from gym_minigrid.wrappers import *
def make_env(env_key, seed=None, blocked=None, num_blocks=None):
    kwargs = {}
    if blocked is not None:
        kwargs["blocked"] = blocked
    if num_blocks is not None:
        kwargs["num_blocks"] = num_blocks
    env = gym.make(env_key, **kwargs)
    # env = RGBImgPartialObsWrapper(env) # Get pixel observations
    # env = ImgObsWrapper(env) # Get rid of the 'mission' field
    # obs = env.reset() # This now produces an RGB tensor only
    env.seed(seed)
    return env