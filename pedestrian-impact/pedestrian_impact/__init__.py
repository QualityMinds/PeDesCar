import logging
from gymnasium.envs.registration import register
from .envs import *

logger = logging.getLogger(__name__)

register(
    id='HumanoidCollision-v0',
    entry_point='pedestrian_impact.envs:HumanoidCollision',
)

register(
    id='HoldHumanoid-v0',
    entry_point='pedestrian_impact.envs:HoldHumanoid',
)

register(
    id='ImitateExpert-v0',
    entry_point='pedestrian_impact.envs:ImitateExpert',
)

register(
    id='ImitateExpertVis-v0',
    entry_point='pedestrian_impact.envs:ImitateExpertVis',
)
