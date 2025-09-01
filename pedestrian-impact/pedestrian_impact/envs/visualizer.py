import os
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np
from mujoco import viewer

from .locomujoco_car_impact import HumanoidCollision
from .utils import data_utils

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


# Adapted from https://github.com/Khrylx/RFC/blob/main/khrylib/rl/utils/visualizer.py
class VisualizerBase:
    def __init__(self, env):
        self.env = env
        self.fr = 0
        self.num_fr = 0
        self.T_arr = [1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60]
        self.T = 8
        self.paused = False
        self.reverse = False
        self.repeat = False
        self.data_gen = self.data_generator()
        self.data = next(self.data_gen)
        super()

    def key_callback(self, key):
        # if key != glfw.KEY_ESCAPE:
        #     return False
        if key == glfw.KEY_DOWN:  # slower play (framerate visualization - like FPS in video)
            self.T = self.T_arr[(self.T_arr.index(self.T) + 1) % len(self.T_arr)]
            print(f'FPS: {self.T}', end=" - ")
        elif key == glfw.KEY_UP:  # faster play (framerate visualization - like FPS in video)
            self.T = self.T_arr[(self.T_arr.index(self.T) - 1) % len(self.T_arr)]
            print(f'FPS: {self.T}', end=" - ")
        elif key == glfw.KEY_KP_ADD:
            self.idx += 1
            self.idx = self.idx % self._num_idxs
            self.env.env.stop()
            self.set_environment()
            self.update_environment = True
            print(f'Reproduce next sequence: ', end=" - ")
        elif key == glfw.KEY_KP_SUBTRACT:
            self.idx -= 1
            self.idx = self._num_idxs - 1 if self.idx < 0 else self.idx
            self.env.env.stop()
            self.set_environment()
            self.update_environment = True
            print(f'Reproduce previous sequence: ', end=" - ")
        elif key == glfw.KEY_KP_1:  # go to the START sequence (last frame)
            self.fr = 0
            self.update_pose()
            print(f'go to first: ', end=" - ")
        elif key == glfw.KEY_KP_2:  # go to the END sequence (first frame)
            self.fr = self.num_fr - 1
            self.update_pose()
            print(f'go to last: ', end=" - ")
        elif key == glfw.KEY_KP_4:  # repeat the sequence
            self.repeat = not self.repeat
            self.update_pose()
            print(f'repeat sequence: ', end=" - ")
        elif key == glfw.KEY_KP_9:  # go in reverse direction (play back)
            self.reverse = not self.reverse
            print(f'play back: ', end=" - ")
        elif key == glfw.KEY_RIGHT:  # move one step backward
            if self.fr < self.num_fr - 1:
                self.fr += 1
            self.update_pose()
            print(f'step backward: ', end=" - ")
        elif key == glfw.KEY_LEFT:  # move one step forward
            if self.fr > 0:
                self.fr -= 1
            self.update_pose()
            print(f'step forward: ', end=" - ")
        elif key == glfw.KEY_SPACE:  # pause
            self.paused = not self.paused
            print(f'paused: {self.paused}')
        else:
            return False
        print(f'|||   current idx: {self.idx}  -  frame: {self.timestamps[self.fr]}')
        return True


# Adapted from https://github.com/Khrylx/RFC/blob/main/khrylib/rl/utils/visualizer.py#L6
class Visualizer(VisualizerBase):

    def __init__(self, path, init_idx, focus=True, render_mode="human"):
        self.files = Path(path)
        self._num_idxs = len(list(self.files.joinpath("files").glob("*.json")))
        self.idx = init_idx
        self.focus = focus
        self.render_mode = render_mode
        net = None  # include agent here.
        self.net = net
        self.set_environment()
        self.update_environment = False

    def set_environment(self):
        data = data_utils.get_mlarki_data_from_json(self.files, self.idx, "joint", "qpos")
        self.mlarki_qpos = data["data"]
        meta = data["meta"]
        self.joints = data["joints"]

        skeleton = meta.data.meta_data_full[0].humanoid_mode
        activity_type = meta.data.meta_data_full[0].activity_type
        use_muscles = True if "muscle" in skeleton.lower() else False
        car_model = meta.data.meta_data_full[0].car_type
        timestep = meta.data.meta_data_full[0].simluation_timestep
        if hasattr(meta.data.meta_data_full[0], "use_foot_forces"):
            use_foot_forces = meta.data.meta_data_full[0].use_foot_forces
        else:
            use_foot_forces = False
        wrapper = HumanoidCollision(env_name=f'HumanCollision{skeleton}.{activity_type}',
                                    car_model=car_model,
                                    use_foot_forces=use_foot_forces,
                                    use_muscles=use_muscles,
                                    disable_arms=False,
                                    timestep=timestep,
                                    std=0.5)
        super().__init__(wrapper)
        idx = wrapper.env._model.body("pelvis").id
        wrapper.env._model.body_pos[idx][0] = meta.data.meta_data_env.human_pos
        wrapper.env._model.body_quat[idx] = meta.data.meta_data_env.human_quat
        # idx = wrapper.env._model.body("car").id
        # wrapper.env._model.body_pos[idx][:2] = meta.data.meta_data_env.car_pos[::-1]
        # ngeom = wrapper.env._model.ngeom - 1
        # self.env_vis.model.geom_rgba[ngeom + 21: ngeom * 2 + 21] = np.array([0.7, 0.0, 0.0, 1])
        # self.env_vis.mujoco_renderer.viewer.cam.lookat[2] = 1.0
        # self.env_vis.mujoco_renderer.viewer.cam.azimuth = 45  # change this in the argument.
        # self.env_vis.mujoco_renderer.viewer.cam.elevation = -8.0
        # self.env_vis.mujoco_renderer.viewer.cam.distance = 5.0

    def lauch(self, show_left_ui=True, show_right_ui=True):
        self.t = 0
        m, d = self.env.env._model, self.env.env._data
        mujoco.mj_forward(m, d)
        self.update_pose()
        # mujoco.mj_resetData(m, d)
        with viewer.launch_passive(m, d,
                                   show_left_ui=show_left_ui,
                                   show_right_ui=show_right_ui,
                                   key_callback=self.key_callback) as viewer_int:

            viewer_int.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer_int.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            # viewer_int.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            m.vis.scale.contactwidth = 0.1
            # m.vis.scale.contactheight = 5.
            m.vis.scale.forcewidth = 0.05
            m.vis.map.force = 0.05

            viewer_int.cam.azimuth = 120  # for paper 120 - original 175
            viewer_int.cam.lookat = (1, 1, 0)  # for paper - original (0,0) or comment
            viewer_int.cam.distance = 6.  # 6.5
            viewer_int.cam.elevation = -25.  # -25

            while viewer_int.is_running():
                if self.t >= np.floor(self.T):
                    if not self.reverse:
                        if self.fr < self.num_fr - 1:
                            self.fr += 1
                        elif self.repeat:
                            self.fr = 0
                    elif self.reverse and self.fr > 0:
                        self.fr -= 1
                    self.update_pose()
                    self.t = 0
                if not self.paused:
                    self.t += 1
                mujoco.mj_forward(m, d)
                viewer_int.sync()
                time.sleep(m.opt.timestep)
                if self.update_environment:
                    viewer_int.close()
                    break
        if self.update_environment:
            self.update_environment = False
            self.lauch(show_left_ui=show_left_ui, show_right_ui=show_right_ui)

    def data_generator(self):
        self.timestamps = np.unique(self.mlarki_qpos.index.get_level_values(0))
        self.num_fr = len(self.timestamps)
        while True:
            yield self.mlarki_qpos

    def update_pose(self):
        for jnt in self.joints:
            data = self.data.loc[self.timestamps[self.fr], jnt].dropna().values
            self.env.env._data.joint(jnt).qpos = data
        # if self.focus:
        #     self.env_vis.mujoco_renderer.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
