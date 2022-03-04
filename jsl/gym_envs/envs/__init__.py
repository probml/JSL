from gym.envs.registration import register

from . import base
from .classification_env import ClassificationEnv
from .regression_env import RegressionEnv

register(
    id="seqcls-v0",
    entry_point="jsl.gym_envs.envs.classification_env:ClassificationEnv")

register(
    id="seqreg-v0",
    entry_point="jsl.gym_envs.envs.regression_env:RegressionEnv")