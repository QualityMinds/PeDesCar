from pedestrian_impact.envs.visualizer import Visualizer
from pedestrian_impact.envs.locomujoco_car_impact import *
from pedestrian_impact.envs.locomujoco_car_impact import HumanoidCollision
from .environments import LocoEnv
from .utils import data_utils

def get_all_task_names():
    return LocoEnv.get_all_task_names()


def download_all_datasets():
    return LocoEnv.download_all_datasets()
