import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import yaml
from ..locomujoco_car_impact import *
import mujoco
import json


def get_skeleton(skeleton_type):
    skeleton_type = skeleton_type.lower()
    if skeleton_type == "attention":
        skeleton: List[Tuple[str, str]] = [
            ("head", "jugularis"),
            ("jugularis", "neck"),
            ("jugularis", "neck"),

            ("l_hip", "l_knee"),
            ("r_hip", "r_knee"),
            ("l_knee", "l_ankle"),
            ("r_knee", "r_ankle"),
            ("pelvis", "r_hip"),
            ("pelvis", "l_hip"),
            ("pelvis", "lower_spine"),
            ("lower_spine", "mid_spine"),
            ("mid_spine", "neck"),
            ("sternum", "neck"),
            ("sternum", "mid_spine"),
            ("neck", "l_shoulder"),
            ("neck", "r_shoulder"),
            ("l_shoulder", "l_elbow"),
            ("r_shoulder", "r_elbow"),
            ("l_elbow", "l_wrist"),
            ("r_elbow", "r_wrist"),
            ("r_knee", "r_lower_leg"),
            ("l_knee", "l_lower_leg"),

            ("l_wrist", "l_lower_arm"),
            ("l_elbow", "l_lower_arm"),
            ("r_wrist", "r_lower_arm"),
            ("r_elbow", "r_lower_arm"),
            ("l_elbow", "l_upper_arm"),
            ("l_shoulder", "l_upper_arm"),
            ("r_elbow", "r_upper_arm"),
            ("r_shoulder", "r_upper_arm"),
        ]
    elif skeleton_type == "mujoco":
        skeleton = None
    elif skeleton_type == "h36m":
        skeleton = None
    elif skeleton_type == "amass" or skeleton_type == "3dpw":
        skeleton = None
    else:
        ValueError("The skeleton format was not found. choose: attention or mujoco.")
    return skeleton


###################################################
################## Wrapper Class ##################
###################################################
class JointConversor():
    def __init__(self, attention_joints=None, mujoco_joints=None):
        if attention_joints is None and mujoco_joints is None:
            self.original = ['world', 'car', 'rear_right_wheel', 'rear_left_wheel', 'steering_body', 'steering_wheel',
                             'front_right_steer_joint', 'front_right_wheel', 'front_left_steer_joint',
                             'front_left_wheel', 'pelvis', 'pelvis_body', 'femur_r', 'hip_r', 'tibia_r', 'talus_r',
                             'calcn_r', 'toes_r', 'femur_l', 'hip_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l',
                             'torso', 'neck', 'head', 'sternum', 'collar_l', 'collar_r', 'spine', 'humerus_r', 'ulna_r',
                             'radius_r', 'hand_r', 'thumb_distal_r', 'little_distal_r', 'humerus_l', 'ulna_l',
                             'radius_l', 'hand_l', 'thumb_distal_l', 'little_distal_l']
            self.mujoco_paper = ['car', 'pelvis_body', 'tibia_r', 'talus_r', 'calcn_r',
                                 'toes_r', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso',
                                 'neck', 'head', 'spine', 'humerus_r', 'radius_r',
                                 'hand_r', 'thumb_distal_r', 'little_distal_r', 'humerus_l', 'radius_l',
                                 'hand_l', 'thumb_distal_l', 'little_distal_l']
            self.h36m = ["Hips",  # 0
                         "RightUpLeg",  # 1
                         "RightLeg",  # 2, 0
                         "RightFoot",  # 3, 1
                         "RightToeBase",  # 4, 2
                         "Site",  # 5, 3
                         "LeftUpLeg",  # 6
                         "LeftLeg",  # 7, 4
                         "LeftFoot",  # 8, 5
                         "LeftToeBase",  # 9, 6
                         "Site",  # 10, 7
                         "Spine",  # 11
                         "Spine1",  # 12, 8
                         "Neck",  # 13, 9
                         "Head",  # 14, 10
                         "Site",  # 15, 11
                         "LeftShoulder",  # 16
                         "LeftArm",  # 17, 12
                         "LeftForeArm",  # 18, 13
                         "LeftHand",  # 19, 14
                         "LeftHandThumb",  # 20
                         "Site",  # 21, 15
                         "L_Wrist_End",  # 22, 16
                         "Site",  # 23
                         "RightShoulder",  # 24
                         "RightArm",  # 25, 17
                         "RightForeArm",  # 26, 18
                         "RightHand",  # 27, 19
                         "RightHandThumb",  # 28
                         "Site",  # 29, 20
                         "R_Wrist_End",  # 30, 21
                         "Site",  # 31
                         ]
            self.amass = ["tibia_l",  # "LeftKnee" 4
                          "tibia_r",  # "RightKnee" 5
                          "spine",  # "Spine2" 6
                          "calcn_l",  # "LeftAnkle" 7
                          "calcn_r",  # "RightAnkle" 8
                          "sternum",  # "Spine3" 9
                          "toes_l",  # "LeftFoot" 10
                          "toes_r",  # "RightFoot" 11
                          "neck",  # "Neck" 12
                          "collar_l",  # "LeftCollar" 13
                          "collar_r",  # "RightCollar" 14
                          "head",  # "Head" 15
                          "humerus_l",  # "LeftShoulder" 16
                          "humerus_r",  # "RightShoulder" 17
                          "radius_l",  # "LeftElbow" 18
                          "radius_r",  # "RightElbow" 19
                          "hand_l",  # "L_Wrist_End" 20
                          "hand_r",  # "R_Wrist_End" 21
                          ]
        else:
            self.attention_joints = attention_joints
            self.mujoco_joints = mujoco_joints
        self.look_at_table = {"world": {"func": self.get_raw_joints,
                                        "jnt_name": "world"},
                              "car": {"func": self.get_raw_joints,
                                      "jnt_name": "car"},
                              'rear_right_wheel': {"func": self.get_raw_joints,
                                                   "jnt_name": "rear_right_wheel"},
                              'rear_left_wheel': {"func": self.get_raw_joints,
                                                  "jnt_name": "rear_left_wheel"},
                              'steering_body': {"func": self.get_raw_joints,
                                                "jnt_name": "steering_body"},
                              'steering_wheel': {"func": self.get_raw_joints,
                                                 "jnt_name": "steering_wheel"},
                              'front_right_steer_joint': {"func": self.get_raw_joints,
                                                          "jnt_name": "front_right_steer_joint"},
                              'front_right_wheel': {"func": self.get_raw_joints,
                                                    "jnt_name": "front_right_wheel"},
                              'front_left_steer_joint': {"func": self.get_raw_joints,
                                                         "jnt_name": "front_left_steer_joint"},
                              'front_left_wheel': {"func": self.get_raw_joints,
                                                   "jnt_name": "front_left_wheel"},
                              "pelvis": {"func": self.get_raw_joints,
                                         "jnt_name": "pelvis"},
                              "r_elbow": {"func": self.get_raw_joints,
                                          "jnt_name": "ulna_r"},
                              "l_ankle": {"func": self.get_raw_joints,
                                          "jnt_name": "tibia_l"},
                              "r_lower_leg": {"func": self.compute_middle_joints,  # Center
                                              "jnt_name": ["tibia_r", "talus_r"]},
                              "sternum": {"func": self.get_raw_joints,
                                          "jnt_name": "sternum"},
                              "l_hip": {"func": self.get_raw_joints,
                                        "jnt_name": "hip_l"},
                              "r_knee": {"func": self.get_raw_joints,
                                         "jnt_name": "tibia_r"},
                              "l_lower_leg": {"func": self.compute_middle_joints,  # Center
                                              "jnt_name": ["tibia_l", "talus_l"]},
                              "r_ankle": {"func": self.get_raw_joints,
                                          "jnt_name": "tibia_r"},
                              "r_femoral_head": {"func": self.get_raw_joints,
                                                 "jnt_name": "femur_r"},
                              "r_upper_arm": {"func": self.compute_middle_joints,  # Center
                                              "jnt_name": ["humerus_r", "radius_r"]},
                              "r_wrist": {"func": self.get_raw_joints,
                                          "jnt_name": "hand_r"},
                              "r_shoulder": {"func": self.get_raw_joints,
                                             "jnt_name": "humerus_r"},
                              "l_knee": {"func": self.get_raw_joints,
                                         "jnt_name": "tibia_l"},
                              "r_lower_arm": {"func": self.compute_middle_joints,  # center
                                              "jnt_name": ["radius_r", "hand_r"]},
                              "l_shoulder": {"func": self.get_raw_joints,
                                             "jnt_name": "humerus_l"},
                              "r_hip": {"func": self.get_raw_joints,
                                        "jnt_name": "hip_r"},
                              "l_upper_arm": {"func": self.compute_middle_joints,  # Center
                                              "jnt_name": ["humerus_l", "radius_l"]},
                              "l_elbow": {"func": self.get_raw_joints,
                                          "jnt_name": "ulna_l"},
                              "l_wrist": {"func": self.get_raw_joints,
                                          "jnt_name": "hand_l"},
                              "l_femoral_head": {"func": self.get_raw_joints,
                                                 "jnt_name": "femur_l"},
                              "l_upper_leg": {"func": self.compute_middle_joints,  # Center
                                              "jnt_name": ["hip_l", "tibia_l"]},
                              "r_upper_leg": {"func": self.compute_middle_joints,  # Center
                                              "jnt_name": ["hip_r", "tibia_r"]},
                              "l_lower_arm": {"func": self.compute_middle_joints,  # center
                                              "jnt_name": ["radius_l", "hand_l"]},
                              "neck": {"func": self.get_raw_joints,
                                       "jnt_name": "neck"},
                              "head": {"func": self.get_raw_joints,
                                       "jnt_name": "head"},
                              "spine": {"func": self.get_raw_joints,
                                        "jnt_name": "spine"},
                              "pelvis_body": {"func": self.get_raw_joints,
                                              "jnt_name": "pelvis_body"},
                              "lower_spine": {"func": self.get_raw_joints,
                                              "jnt_name": "torso"},
                              "mid_spine": {"func": self.compute_middle_joints,  # center
                                            "jnt_name": ["torso", ["humerus_r", "humerus_l"]]},
                              "jugularis": {"func": self.compute_middle_joints,  # center
                                            "jnt_name": ["neck", ["humerus_r", "humerus_l"]]},
                              }

    def compute_middle_joints(self, **kwargs):
        data = getattr(kwargs["data"], kwargs["attrib"])
        data_vals = []
        for jnt in kwargs["jnt_name"]:
            if isinstance(jnt, str):
                vals = data[kwargs["data"].body(jnt).id]
            else:
                vals = self.compute_middle_joints(data=kwargs["data"], jnt_name=jnt, attrib=kwargs["attrib"])
            data_vals.append(vals)
        return np.mean(data_vals, 0)

    def get_raw_joints(self, **kwargs):
        return getattr(kwargs["data"], kwargs["attrib"])[kwargs["data"].body(kwargs["jnt_name"]).id]

    def _get_and_map_mujoco_to_attention(self, data_mujoco, jnt, attrib):
        data_attention = self.look_at_table[jnt]["func"](data=data_mujoco,
                                                         jnt_name=self.look_at_table[jnt]["jnt_name"],
                                                         attrib=attrib)
        return data_attention.tolist()


#########################################################
#########################################################
#########################################################
def _load_data_from_folder(path, idx):
    path = Path(path)
    files = list(path.joinpath("files").glob("*.json"))
    files.sort()
    file = files[idx]  # CHANGE FILE ID

    data = load_json(str(file), class_mode=True)
    return data


def get_numpy_data_from_json(path, idx, data_type, joint_type="", exclude_joint=""):
    if data_type == "joint":
        if joint_type == "":
            joint_type = "qpos"
            print("you can choose qvel and qpos for joint type. qpos is by default")
    if Path(path).suffix == "":
        data = _load_data_from_folder(path, idx)
    else:
        data = load_json(str(path), class_mode=True)
    meta_data = data.meta
    nsamples = np.arange(len(data.data))

    joints_of_interest = data.meta.data.joint_names

    # for normal numpy array to compute ranges.
    numpy_data = []
    for timestamp in nsamples:
        object = getattr(data.data[timestamp], data_type)
        pose_data = getattr(object, joint_type)
        numpy_data.append(pose_data)

    numpy_data = np.array(numpy_data)

    return {"data": numpy_data,
            "meta": meta_data,
            "joints": joints_of_interest}


def get_mlarki_data_from_json(path, idx, data_type, joint_type="qpos", exclude_joint=""):
    if data_type == "joint":
        print("you can choose qvel and qpos for joint type. qpos is by default")

    if Path(path).suffix == "":
        data = _load_data_from_folder(path, idx)
    else:
        data = load_json(str(path), class_mode=True)
    meta_data = data.meta
    nsamples = np.arange(len(data.data))

    joints_of_interest = data.meta.data.joint_names

    mlarki_data = []
    cols = ["q"]
    for timestamp in nsamples:
        object = getattr(data.data[timestamp], data_type)
        pose_data = getattr(object, joint_type)
        pose3d = pd.DataFrame(pose_data, columns=cols, index=joints_of_interest)
        pose3d.index.name = "body_part"
        mlarki_data.append(pose3d)

    multi_index = np.array([[str(data.data[a].it).zfill(6), b] for a in nsamples for b in joints_of_interest])
    multi_index = pd.MultiIndex.from_arrays([multi_index[:, 0], multi_index[:, 1]], names=('time', 'body_part'))
    mlarki_data = pd.DataFrame(np.array(mlarki_data).reshape(-1, len(cols)), index=multi_index, columns=cols)

    return {"data": mlarki_data,
            "meta": meta_data,
            "joints": joints_of_interest}


def _make_env(meta, agent_dir="", return_wrapper=False):
    wrapper = HumanoidCollision(env_name=f'HumanCollision{meta.humanoid_mode}.{meta.activity_type}',
                                car_model=meta.car_type,
                                use_foot_forces=meta.use_foot_forces,
                                use_muscles=False if meta.humanoid_mode != "Muscle" else True,
                                disable_arms=False,
                                timestep=meta.simluation_timestep,
                                n_substeps=1,
                                agent_dir=agent_dir,
                                # seed=0,
                                # viewer_params=viewer_params,
                                std=0.5)
    wrapper.env.reset()
    mujoco.mj_resetData(wrapper.env._model, wrapper.env._data)
    mujoco.mj_forward(wrapper.env._model, wrapper.env._data)
    m, d = wrapper.env._model, wrapper.env._data
    if return_wrapper:
        return wrapper, m, d
    else:
        del wrapper
        return m, d


def get_body_names(meta_data, agent_dir="", attention=False):  # ONLY ACCEPTS NUMPY
    meta = meta_data.data.meta_data_full[0]
    _, m, _ = _make_env(meta, agent_dir, return_wrapper=True)
    body_names = [m.body(i).name for i in range(m.nbody)]
    return body_names


def efficient_convert_numpy_qs_to_body(meta_data, numpy_data, m, d, attention=False):  # ONLY ACCEPTS NUMPY
    idx = m.body("pelvis").id
    m.body_pos[idx][0] = meta_data.data.meta_data_env.human_pos
    m.body_quat[idx] = meta_data.data.meta_data_env.human_quat

    conversor = JointConversor()
    nsamples = np.arange(len(numpy_data))
    xpos = []
    for timestamp in nsamples:
        d.qpos = numpy_data[timestamp].copy().tolist()
        mujoco.mj_forward(m, d)
        if attention:
            pose_data = []
            for j in conversor.look_at_table:
                pose_data.append(conversor._get_and_map_mujoco_to_attention(d, j, "xpos"))
        else:
            pose_data = d.xpos.copy()
        # quat = env.env._data.xquat
        xpos.append(pose_data)
    xpos = np.array(xpos)
    del m, d
    return xpos


def convert_mlarki_qs_to_body(meta_data, mlarki_data, agent_dir, attention=False):  # ONLY ACCEPTS NUMPY
    meta = meta_data.data.meta_data_full[0]
    wrapper, m, d = _make_env(meta, agent_dir, return_wrapper=True)
    idx = m.body("pelvis").id
    m.body_pos[idx][0] = meta_data.data.meta_data_env.human_pos
    m.body_quat[idx] = meta_data.data.meta_data_env.human_quat
    conversor = JointConversor()

    if attention:
        joint_names = list(conversor.look_at_table.keys())
    else:
        joint_names = [wrapper.env._model.body(i).name for i in range(wrapper.env._model.nbody)]
    indexes = np.unique(mlarki_data.index.get_level_values(0))
    nsamples = np.arange(len(indexes))
    cols = ["x", "y", "z"]
    xpos = []
    for timestamp in nsamples:
        data = mlarki_data.loc[indexes[timestamp]].values.flatten()
        d.qpos = data.copy()
        mujoco.mj_forward(m, d)
        if attention:
            pose_data = []
            for j in conversor.look_at_table:
                pose_data.append(conversor._get_and_map_mujoco_to_attention(d, j, "xpos"))
        else:
            pose_data = d.xpos.copy()
        pose_data = pd.DataFrame(pose_data, columns=cols, index=joint_names)
        pose_data.index.name = "body_part"
        xpos.append(pose_data)

    multi_index = np.array([[a, b] for a in indexes for b in joint_names])
    multi_index = pd.MultiIndex.from_arrays([multi_index[:, 0], multi_index[:, 1]], names=('time', 'body_part'))
    xpos = pd.DataFrame(np.array(xpos).reshape(-1, len(cols)), index=multi_index, columns=cols)
    return xpos


class Struct(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Struct(b) if isinstance(b, dict) else b)


def RemoveStruct(opt):
    opt1 = {}
    if not isinstance(opt, (dict, Struct)):
        if not isinstance(opt, (int, str, float)):
            opt = opt.tolist()
        return opt
    for k in list(opt.__dict__.keys()):
        if isinstance(getattr(opt, k), Struct):
            opt1[k] = RemoveStruct(getattr(opt, k))
        elif isinstance(getattr(opt, k), list):
            opt_list = []
            for i in range(len(getattr(opt, k))):
                opt_list.append(RemoveStruct(getattr(opt, k)[i]))
            opt1[k] = opt_list
        else:
            opt1[k] = getattr(opt, k)
    return opt1


# Read YAML file
def load_yaml(path, class_mode=False):
    """ A function to load YAML file"""
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
    if class_mode:
        data = Struct(data)
    return data


# Save a json file
def save_json(path, data):
    json_object = json.dumps(data, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)


# Load a json file
def load_json(path, class_mode=False):
    """ A function to load JSON file"""
    with open(path, 'rb') as stream:
        data = json.load(stream)
    if class_mode:
        data = Struct(data)
    return data


# Write YAML file
def write_yaml(data, path="default_output.yaml", remote_struct=True):
    """ A function to write YAML file"""
    if remote_struct:
        data = RemoveStruct(data)
    with open(path, 'w') as f:
        yaml.dump(data, f)
