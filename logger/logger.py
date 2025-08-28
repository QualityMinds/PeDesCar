import copy
import json
from datetime import datetime
from pathlib import Path

import cv2
import glfw
import numpy as np

import pedestrian_impact as env


class Logger(env.data_utils.JointConversor):
    def __init__(self, path, convert_joints_to_attention=True):
        path = Path(path)
        if path.joinpath("dataset_summary.json").exists():
            self.output_dir = path
        else:
            curr_time = datetime.utcnow().strftime('%Y%m%d_%H%M-id%f')[:-2]
            self.output_dir = path.joinpath(f'{curr_time}')
        self.videos_dir = self.output_dir.joinpath("videos")
        self.files_dir = self.output_dir.joinpath("files")
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.simulation = {}
        self.record_simulation_data = {}
        self.videos = {}
        self.convert_joints_to_attention = convert_joints_to_attention
        super().__init__()

    def record_simulation_meta(self, data):
        self.record_simulation_data[f'sim{self.simulation:08d}'] = data

    def _define_bodys(self):
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]  # kinamtics
        # self.human_body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        # self.car_body_names = []
        # for jnt in self.human_body_names.copy():
        #     if "steer" in jnt or "car" in jnt or "wheel" in jnt or "axle" in jnt:
        #         self.human_body_names.remove(jnt)
        #         self.car_body_names.append(jnt)
        self.human_joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]  # kinamtics
        self.car_joint_names = []
        for jnt in self.human_joint_names.copy():
            if "steer" in jnt or "car" in jnt or "wheel" in jnt or "axle" in jnt:
                self.human_joint_names.remove(jnt)
                self.car_joint_names.append(jnt)

        # self.human_geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]  # geometry positions
        # self.human_geom_names = [i for i in self.human_geom_names if not "floor" in i]  # remove the floor geom
        # self.car_geom_names = []
        # for jnt in self.human_geom_names.copy():
        #     if "steer" in jnt or (
        #             "car" in jnt and not "metacar" in jnt) or "wheel" in jnt or "axle" in jnt or "chassis" in jnt:
        #         self.human_geom_names.remove(jnt)
        #         self.car_geom_names.append(jnt)

        self.frame = {}

    def update_model(self, model, task):
        self.task = task.split(".")[-1]
        self.model = model
        self._define_bodys()

    def new_frame(self, simulation):
        self.simulation = simulation
        self.frame[f'sim{simulation:08d}'] = {}
        self.frame[f'sim{simulation:08d}']["data"] = []
        meta = {"task": self.task,
                # "video_path": str(self.videos_dir.joinpath(f'{simulation:08d}.mp4')),
                }
        self.frame[f'sim{simulation:08d}']["meta"] = meta

    @staticmethod
    def get_body_geom_positions(data, jnt_name):
        return data.geom_xpos[data.geom(jnt_name).id].tolist()

    @staticmethod
    def get_body_info(data, jnt_name, attrib):
        return getattr(data, attrib)[data.body(jnt_name).id].tolist()

    @staticmethod
    def get_joint_info(data, jnt_name, attrib):
        return getattr(data.joint(jnt_name), attrib).tolist()

    def add_frame(self, it, data):
        template = {"it": it,
                    "time": np.round(data.time, 10),  # seconds
                    "joint": {},
                    }
        template["joint"]["qpos"] = data.qpos.tolist()
        template["joint"]["qvel"] = data.qvel.tolist()
        self.frame[f'sim{self.simulation:08d}']["data"].append(copy.deepcopy(template))
        del template

    def add_video(self, simulation, video):
        name = list(video.keys())[0]
        if len(video[name]) > 0:
            self.videos[f'sim{simulation:08d}'] = video

    def add_meta_data(self, meta):
        meta["joint_names"] = self.joint_names
        self.frame[f'sim{self.simulation:08d}']["meta"]["data"] = meta

    def clear_data(self):
        self.frame = {}
        print("frame and data were deleted...")

    def save_json_one_file(self):
        json_object = json.dumps(self.frame, indent=4)
        with open(str(self.output_dir.joinpath(f'dataset.json')), "w") as outfile:
            outfile.write(json_object)

    def save_json_multiple_files(self):
        summary = self.output_dir.joinpath(f'dataset_summary.json')
        if summary.exists():
            with open(str(summary), 'r') as file:
                json_object = json.load(file)
            json_object.update(self.record_simulation_data)
            json_object = json.dumps(json_object, indent=4)
        else:
            json_object = json.dumps(self.record_simulation_data, indent=4)
        with open(str(summary), "w") as outfile:
            outfile.write(json_object)
        for key, obj in self.frame.items():
            if len(self.videos) > 0:
                cameras = list(self.videos[key].keys())
                sz = self.videos[key][cameras[0]][0].shape[:2][::-1]
                for cam, video in self.videos[key].items():
                    video_name = f'{key}_{cam}.mp4'  # .replace("sim", "vid")
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    out = cv2.VideoWriter(str(self.videos_dir.joinpath(video_name)), fourcc, 30.0, sz)
                    for img in video:
                        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    out.release()
            json_object = json.dumps(obj, indent=4)
            with open(str(self.files_dir.joinpath(f'{key}.json')), "w") as outfile:
                outfile.write(json_object)

    def clean_data(self):
        self.frame.clear()
        self.videos.clear()
        self.record_simulation_data.clear()
