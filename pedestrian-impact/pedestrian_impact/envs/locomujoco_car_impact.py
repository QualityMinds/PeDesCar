import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from mushroom_rl.core import Agent, Environment
from scipy.spatial.transform import Rotation as R

from .environments import LocoEnv

torch.manual_seed(0)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def euler_to_quaternion(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    return r.as_quat()


def quaternion_to_euler(quat, degrees=False):
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=degrees)


def kmh2ms(speed_kmh):
    return speed_kmh * 1000 / 3600


def ms2kmh(speed_ms):
    return speed_ms * 3600 / 1000


def get_emi_pose(pose_type):
    import pickle
    assert pose_type in ["gait_inv", "gait", "group_1", "group_2",
                         "group_3"], f"{pose_type} - This pose type is not valid. Valid options are: gait_inv, gait, group_1, group_2, group_3"
    with open(Path(CURR_DIR).joinpath("assets", "emi_data", "emi_q_poses.pkl"), 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data[pose_type]


class HumanoidCollision(LocoEnv):
    render_mode = "human"

    def __init__(self, env_name, car_model, *args, **kwargs):

        if '.' in env_name:
            env_data = env_name.split('.')
            env_name = env_data[0]
            if len(env_data) == 2:
                idx = 1
            else:
                idx = 2
            args = [env_data[idx]] + list(args)
        env = Environment._registered_envs[env_name]
        if env.__name__ == "HumanCollisionTorque4Ages":
            kwargs["mode"] = env_data[idx - 1]
            kwargs["n_models"] = 1

        if kwargs.get("agent_path"):
            if isinstance(kwargs["agent_path"], str) and kwargs["agent_path"] != "":
                model_path = kwargs["agent_path"]
            else:
                raise NameError("input format my agent is not string. Path to Agent model")
        else:
            env_name_int = f'{env_data[0]}'.replace("HumanCollision", "Humanoid")
            if idx == 1:
                env_name_int = f'{env_name_int}.{env_data[idx]}'
            elif idx == 2:
                env_name_int = f'{env_name_int}.{env_data[idx]}.{env_data[idx - 1]}'
            CURR_DIR = os.path.dirname(os.path.abspath(__file__))
            if kwargs.get("agent_dir"):
                if isinstance(kwargs["agent_dir"], str) and kwargs["agent_dir"] != "":
                    CURR_DIR = kwargs["agent_dir"]
            available_models = list(Path(CURR_DIR).joinpath("agents/", env_name_int).rglob("*.msh"))
            available_models = [m for m in available_models if not "use_foot_forces___True" in str(m)]
            if len(available_models) == 1:
                model_path = available_models[0]
            elif len(available_models) == 0:
                print(f'current path: {Path(CURR_DIR).joinpath("agents/", env_name_int)}')
                raise FileExistsError("No files found")
            else:
                Jvalues = [float(str(m)[str(m).find("_J_") + 3:str(m).find(".msh")]) for m in available_models]
                best_model = np.argmax(Jvalues)
                model_path = available_models[best_model]
        print(f'model:  {model_path}\n\n')
        self.agent = Agent.load(model_path)
        self.reset_agent()
        del kwargs["std"]
        agent_path = kwargs.get('agent_path')
        if agent_path is not None: del kwargs["agent_path"]
        if kwargs.get("agent_dir"): del kwargs["agent_dir"]
        kwargs["car_model"] = car_model
        if hasattr(env, 'generate'):
            self.env = env.generate(*args, **kwargs)
        else:
            self.env = env(*args, **kwargs)
        if idx == 2:
            self.env.height_bound = self.env.has_fallen_scale[kwargs["mode"]]
        # print("environment and agent were loaded correctly...")

    def reset_agent(self):
        """
        Reset the state of the agent.
        """
        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps = 0

    def create_new_dataset(self):
        self.dataset = self.env.create_dataset()

    def play_trajectory(self, n_steps_per_episode, n_episodes=None):
        self.env.play_trajectory_from_velocity(n_steps_per_episode=n_steps_per_episode, n_episodes=n_episodes)

    def verify_collision_w_car(self):
        datai = self.env._data
        collision_speed = 0
        activate_agent = True
        for coni in range(datai.ncon):
            con = datai.contact[coni]
            if con.geom1 == 1 or con.geom2 == 1:
                activate_agent = False
                collision_speed = ms2kmh(self.env._data.joint("car_tx").qvel[0])
                break
        return activate_agent, collision_speed

    def init_env(self,
                 init_car_speed=[5, 65],  # [m]
                 init_car_turn=[-0.35, 0.35],  # [rads]???
                 init_car_position=[-5, 0],  # [m]
                 init_car_position_shift=[-0.4, 0.4],  # [m]
                 init_human_position=[-2, -0],  # [m]
                 init_human_orientation=[-45, 45],  # [degrees]
                 flip_human=True,
                 ):
        """
        Range of the initial values that will be used to set the car and human physics. Consider the following:
        - Collisions happens at X=0,Y=0.
        - Qvel speed is an estimation.
        - ...
        init_car_speed: speed applied to the qvel of the chasis to meet the desired speed (km/h)
        init_car_turn: turn applied to the qvel of the wheels for left/right rotations
        init_car_position: position of the car in world coordinates in the X-axis (in meters). move direction axis.
        init_car_position_shift: shift position of the car in world in the Y-axis (in meters). Perpendiular to the movement direction.
        init_human_position: position of the human in world coordinates in the X-axis (in meters). move direction axis.
        """
        observation = self.env.reset()

        idx = self.env._model.body("pelvis").id
        pos = np.random.uniform(init_human_position[0], init_human_position[1])
        roll = np.random.uniform(init_human_orientation[0], init_human_orientation[1])
        val180 = 180
        if flip_human:
            rotate = np.random.choice([0, 180])
            roll += rotate
            pos = -pos if abs(roll) > val180 / 2 else pos
        # roll -= val180
        quat = euler_to_quaternion(*[roll, 0, 90])
        if np.sum(init_human_position) != 0:
            self.env._model.body_pos[idx][0] = pos
        if np.sum(init_human_orientation) != 0:
            self.env._model.body_quat[idx] = quat

        ########### Car movement ###########
        turn = np.round(np.random.uniform(init_car_turn[0], init_car_turn[1]), 6)
        self.env._data.joint("front_right_steer_joint").qpos = turn
        self.env._data.joint("front_left_steer_joint").qpos = turn
        self.env._data.joint("steering_joint").qvel = turn
        speed = np.round(np.random.uniform(kmh2ms(init_car_speed[0]), kmh2ms(init_car_speed[1])), 6)
        # speed = np.round(np.random.uniform(kmh2ms(5), kmh2ms(15)), 6)
        self.env._data.joint("rear_left_wheel_joint").qvel = speed
        self.env._data.joint("rear_right_wheel_joint").qvel = speed
        self.env._data.joint("car_tx").qvel = speed

        car_pos_y = np.round(np.random.uniform(init_car_position_shift[0], init_car_position_shift[1]), 6)

        # pp = np.random.choice(["gait_inv", "gait", "group_1", "group_2", "group_3"])
        # print(pp)s
        # data = get_emi_pose("gait")
        # for jnt in data.index:
        #     self.env._data.joint(jnt).qpos = data.loc[jnt]
        # mujoco.mj_forward(self.env._model, self.env._data)

        scale = (init_car_position[0] - init_car_position[1]) / (init_car_speed[1] - init_car_speed[0])
        max_pos = scale * (ms2kmh(speed) - init_car_speed[0]) + init_car_position[1] + 1e-8
        car_pos_x = np.round(np.random.uniform(max_pos, init_car_position[1]), 6)
        self.env._data.joint("car_tx").qpos = car_pos_x
        self.env._data.joint("car_ty").qpos = car_pos_y
        return observation, {"human_pos": pos,
                             "human_rot": roll,
                             "human_quat": quat.tolist(),  # same as roll but in quaternion,
                             "wheel_speed": speed,
                             "turn": turn,
                             "car_pos": [car_pos_x, car_pos_y],
                             "car_speed": speed}


if __name__ == "__main__":
    # THIS MAIN FUNCTION IS NOT VALIDATED FOR THE LAST ENV VERSION =(
    print("no updated code here")
