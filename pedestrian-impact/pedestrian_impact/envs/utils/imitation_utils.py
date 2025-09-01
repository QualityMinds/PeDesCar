# Adapted from https://github.com/Khrylx/RFC/blob/main/motion_imitation/utils/tools.py
import numpy as np
import torch
from transforms3d import _gohlketransforms


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, clip=10.0):
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        x = (x - self.rs.mean) / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def set_mean_std(self, mean, std, n):
        self.rs._n = n
        self.rs._M[...] = mean
        self.rs._S[...] = std


def multi_quat_norm(nq):
    """return the scalar rotation of a N joints"""
    nq_norm = np.arccos(np.clip(abs(nq[::4]), -1.0, 1.0))
    return nq_norm


def multi_quat_diff(nq1, nq0):
    """return the relative quaternions q1-q0 of N joints"""
    nq_diff = np.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4 * i, 4 * i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        nq_diff[ind] = _gohlketransforms.quaternion_multiply(q1, _gohlketransforms.quaternion_inverse(q0))
    return nq_diff


def rotation_from_quaternion(quaternion, separate=False):
    if 1 - abs(quaternion[0]) < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
    else:
        s = np.sqrt(1 - quaternion[0] * quaternion[0])
        axis = quaternion[1:4] / s
        angle = 2 * np.arccos(quaternion[0])
    return (axis, angle) if separate else axis * angle


def get_body_qposaddr(model, body_names):
    body_qposaddr = dict()
    for i, body_name in enumerate(body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr


def align_human_state(qpos, qvel, ref_qpos):
    qpos[:2] = ref_qpos[:2]
    hq = get_heading_q(ref_qpos[3:7])
    qpos[3:7] = _gohlketransforms.quaternion_multiply(hq, qpos[3:7])
    qvel[:3] = quat_mul_vec(hq, qvel[:3])


def get_traj_pos(orig_traj):
    traj_pos = orig_traj[:, 2:].copy()
    for i in range(traj_pos.shape[0]):
        traj_pos[i, 1:5] = de_heading(traj_pos[i, 1:5])
    return traj_pos


def get_traj_vel(orig_traj, dt):
    traj_vel = []
    for i in range(orig_traj.shape[0] - 1):
        vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], dt, 'heading')
        traj_vel.append(vel)
    traj_vel.append(traj_vel[-1].copy())
    traj_vel = np.vstack(traj_vel)
    return traj_vel


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * np.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_qvel_fd(cur_qpos, next_qpos, dt, transform=None):
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    qrel = _gohlketransforms.quaternion_multiply(next_qpos[3:7], _gohlketransforms.quaternion_inverse(cur_qpos[3:7]))
    axis, angle = rotation_from_quaternion(qrel, True)
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    rv = (axis * angle) / dt
    rv = transform_vec(rv, cur_qpos[3:7], 'root')  # angular velocity is in root coord
    qvel = (next_qpos[7:] - cur_qpos[7:]) / dt
    qvel = np.concatenate((v, rv, qvel))
    if transform is not None:
        v = transform_vec(v, cur_qpos[3:7], transform)
        qvel[:3] = v
    return qvel


def get_qvel_fd_new(cur_qpos, next_qpos, dt, transform=None):
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    qrel = _gohlketransforms.quaternion_multiply(next_qpos[3:7], _gohlketransforms.quaternion_inverse(cur_qpos[3:7]))
    axis, angle = rotation_from_quaternion(qrel, True)
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    rv = (axis * angle) / dt
    rv = transform_vec(rv, cur_qpos[3:7], 'root')  # angular velocity is in root coord
    diff = next_qpos[7:] - cur_qpos[7:]
    while np.any(diff > np.pi):
        diff[diff > np.pi] -= 2 * np.pi
    while np.any(diff < -np.pi):
        diff[diff < -np.pi] += 2 * np.pi
    qvel = diff / dt
    qvel = np.concatenate((v, rv, qvel))
    if transform is not None:
        v = transform_vec(v, cur_qpos[3:7], transform)
        qvel[:3] = v
    return qvel


def get_angvel_fd(prev_bquat, cur_bquat, dt):
    q_diff = multi_quat_diff(cur_bquat, prev_bquat)
    n_joint = q_diff.shape[0] // 4
    body_angvel = np.zeros(n_joint * 3)
    for i in range(n_joint):
        body_angvel[3 * i: 3 * i + 3] = rotation_from_quaternion(q_diff[4 * i: 4 * i + 4]) / dt
    return body_angvel


def transform_vec(v, q, trans='root'):
    if trans == 'root':
        rot = _gohlketransforms.quaternion_matrix(q)[:3, :3]
    elif trans == 'heading':
        hq = q.copy()
        hq[1] = 0
        hq[2] = 0
        hq /= np.linalg.norm(hq)
        rot = _gohlketransforms.quaternion_matrix(hq)[:3, :3]
    else:
        assert False
    v = rot.T.dot(v[:, None]).ravel()
    return v


def get_heading_q(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    hq /= np.linalg.norm(hq)
    return hq


def get_heading(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    if hq[3] < 0:
        hq *= -1
    hq /= np.linalg.norm(hq)
    return 2 * np.arccos(hq[0])


def de_heading(q):
    return _gohlketransforms.quaternion_multiply(_gohlketransforms.quaternion_inverse(get_heading_q(q)), q)


def multi_quat_diff(nq1, nq0):
    """return the relative quaternions q1-q0 of N joints"""

    nq_diff = np.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4 * i, 4 * i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        nq_diff[ind] = _gohlketransforms.quaternion_multiply(q1, _gohlketransforms.quaternion_inverse(q0))
    return nq_diff


def multi_quat_norm(nq):
    """return the scalar rotation of a N joints"""

    nq_norm = np.arccos(np.clip(abs(nq[::4]), -1.0, 1.0))
    return nq_norm


def quat_mul_vec(q, v):
    old_shape = v.shape
    v = v.reshape(-1, 3)
    v = v.dot(_gohlketransforms.quaternion_matrix(q)[:3, :3].T)
    return v.reshape(old_shape)


def quat_to_bullet(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_from_bullet(q):
    return np.array([q[3], q[0], q[1], q[2]])


def quat_from_expmap(e):
    angle = np.linalg.norm(e)
    if angle < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = e / angle
    return _gohlketransforms.quaternion_about_axis(angle, axis)
