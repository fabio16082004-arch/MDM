import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
))
from dataset_index import recover_joints, root_trajectory, FID_L, FID_R

from utils.rotation_conversions import (
    rotation_6d_to_matrix, matrix_to_rotation_6d,
    matrix_to_quaternion, quaternion_to_matrix,
)

ROT6D_SLICE = slice(67, 193)
JOINT_VEL_SLICE = slice(193, 259)

DECEL_FRAMES_FIXED = 10
UNIFORM_TARGET_LENGTH = 80


def _quaternions_from_frame(frame_vec263):
    rot6d = torch.from_numpy(frame_vec263[ROT6D_SLICE].reshape(21, 6).copy()).float()
    mat = rotation_6d_to_matrix(rot6d)
    return matrix_to_quaternion(mat)


def _slerp_quaternions(q0, q1, t):
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    lerp_result = (1 - t) * q0 + t * q1
    lerp_result = lerp_result / lerp_result.norm(dim=-1, keepdim=True)
    coef0 = torch.sin((1 - t) * omega) / sin_omega.clamp(min=1e-8)
    coef1 = torch.sin(t * omega) / sin_omega.clamp(min=1e-8)
    slerp_result = coef0 * q0 + coef1 * q1
    small_angle = sin_omega < 1e-6
    return torch.where(small_angle, lerp_result, slerp_result)


def ric_index_for_joint(joint_idx_in_22):
    return joint_idx_in_22 - 1


def _foot_height_ric(frame_vec263):
    foot_joint_indices = [FID_L[0], FID_L[1], FID_R[0], FID_R[1]]
    heights = []
    for j in foot_joint_indices:
        ric_idx = ric_index_for_joint(j)
        y_col = 4 + ric_idx * 3 + 1
        heights.append(frame_vec263[y_col])
    return float(np.mean(heights))


def _height_corrected_reference(reference_frame, anchor_frame):
    delta = _foot_height_ric(anchor_frame) - _foot_height_ric(reference_frame)
    corrected = reference_frame.copy()
    for ric_idx in range(21):
        y_col = 4 + ric_idx * 3 + 1
        corrected[y_col] += delta
    return corrected


def foot_arc_height_offset(t, max_lift):
    t = np.clip(t, 0.0, 1.0)
    return max_lift * (1 - np.cos(2 * np.pi * t)) / 2.0


def compute_effective_max_lift(current_frame, reference_frame, foot,
                                max_lift=0.08, lift_ratio=0.35):
    ankle_idx = FID_L[0] if foot == 'left' else FID_R[0]
    ric_idx = ric_index_for_joint(ankle_idx)
    x_col, z_col = 4 + ric_idx * 3, 4 + ric_idx * 3 + 2

    curr_xz = np.array([current_frame[x_col], current_frame[z_col]])
    ref_xz = np.array([reference_frame[x_col], reference_frame[z_col]])
    horizontal_dist = np.linalg.norm(curr_xz - ref_xz)

    return min(max_lift, horizontal_dist * lift_ratio)


def identify_lagging_foot(data, end_frame, lookback=10):
    start = max(0, end_frame - lookback)
    traj = root_trajectory(data[start:end_frame])
    direction = traj[-1] - traj[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    direction = direction / norm

    joints = recover_joints(data[end_frame - 1:end_frame])[0]
    left_pos = joints[FID_L[0], [0, 2]]
    right_pos = joints[FID_R[0], [0, 2]]

    left_proj = np.dot(left_pos, direction)
    right_proj = np.dot(right_pos, direction)

    return 'left' if left_proj < right_proj else 'right'


def interpolate_frame_towards_reference(current_frame, reference_frame, t,
                                         lagging_foot=None, effective_max_lift=0.0):
    out = current_frame.copy()

    out[0:3] = (1 - t) * current_frame[0:3]
    out[3] = current_frame[3]

    out[4:67] = (1 - t) * current_frame[4:67] + t * reference_frame[4:67]

    if lagging_foot is not None and effective_max_lift > 1e-6:
        ankle_joint_idx = FID_L[0] if lagging_foot == 'left' else FID_R[0]
        ric_idx = ric_index_for_joint(ankle_joint_idx)
        y_col = 4 + ric_idx * 3 + 1
        out[y_col] += foot_arc_height_offset(t, effective_max_lift)

        toe_joint_idx = FID_L[1] if lagging_foot == 'left' else FID_R[1]
        ric_idx_toe = ric_index_for_joint(toe_joint_idx)
        y_col_toe = 4 + ric_idx_toe * 3 + 1
        out[y_col_toe] += foot_arc_height_offset(t, effective_max_lift)

    q_curr = _quaternions_from_frame(current_frame)
    q_ref = _quaternions_from_frame(reference_frame)
    t_tensor = torch.tensor(float(t))
    q_interp = _slerp_quaternions(q_curr, q_ref, t_tensor)
    mat_interp = quaternion_to_matrix(q_interp)
    out[ROT6D_SLICE] = matrix_to_rotation_6d(mat_interp).reshape(-1).numpy()

    out[JOINT_VEL_SLICE] = (1 - t) * current_frame[JOINT_VEL_SLICE]
    out[259:263] = (1 - t) * current_frame[259:263] + t * reference_frame[259:263]

    return out


def build_slerp_tail(data, end_frame, reference_frame, class_L=None,
                      decel_frames=None, quiet_seconds=0.5, fps=20,
                      total_tail_frames=None,
                      noise_std=0.001, use_foot_arc=True, max_lift=0.08,
                      lift_ratio=0.35, lookback=10):
    last_frame = data[end_frame - 1].astype(np.float32)

    if total_tail_frames is not None:
        decel_frames = min(DECEL_FRAMES_FIXED, max(1, total_tail_frames))
        quiet_frames_override = max(0, total_tail_frames - decel_frames)
    else:
        quiet_frames_override = None
        if decel_frames is None:
            decel_frames = DECEL_FRAMES_FIXED

    lagging_foot = identify_lagging_foot(data, end_frame, lookback=lookback) \
        if use_foot_arc else None

    effective_max_lift = 0.0
    if lagging_foot is not None:
        effective_max_lift = compute_effective_max_lift(
            last_frame, reference_frame, lagging_foot,
            max_lift=max_lift, lift_ratio=lift_ratio,
        )

    quiet_frames = quiet_frames_override if quiet_frames_override is not None else int(quiet_seconds * fps)
    total_frames = decel_frames + quiet_frames
    tail = np.zeros((total_frames, 263), dtype=np.float32)

    for i in range(decel_frames):
        t = (i + 1) / decel_frames
        tail[i] = interpolate_frame_towards_reference(
            last_frame, reference_frame, t,
            lagging_foot=lagging_foot, effective_max_lift=effective_max_lift
        )

    last_decel_frame = tail[decel_frames - 1] if decel_frames > 0 else last_frame

    reference_frame_corrected = _height_corrected_reference(reference_frame, last_decel_frame)

    for i in range(quiet_frames):
        frame = reference_frame_corrected.copy()
        frame[0:3] = 0.0
        frame[3] = last_decel_frame[3]

        tail[decel_frames + i] = frame

    return tail


def build_slerp_head(data, start_frame, reference_frame, class_L=None,
                      accel_frames=None, quiet_seconds=0.5, fps=20,
                      total_head_frames=None,
                      noise_std=0.001, use_foot_arc=True, max_lift=0.08,
                      lift_ratio=0.35, lookback=10):
    first_frame = data[start_frame].astype(np.float32)

    if total_head_frames is not None:
        accel_frames = min(DECEL_FRAMES_FIXED, max(1, total_head_frames))
        quiet_frames_override = max(0, total_head_frames - accel_frames)
    else:
        quiet_frames_override = None
        if accel_frames is None:
            accel_frames = DECEL_FRAMES_FIXED

    effective_lookback = min(lookback, len(data) - start_frame - 1)
    leading_foot = None
    if use_foot_arc and effective_lookback > 1:
        leading_foot = identify_lagging_foot(data, start_frame + effective_lookback,
                                              lookback=effective_lookback)

    effective_max_lift = 0.0
    if leading_foot is not None:
        effective_max_lift = compute_effective_max_lift(
            first_frame, reference_frame, leading_foot,
            max_lift=max_lift, lift_ratio=lift_ratio,
        )

    quiet_frames = quiet_frames_override if quiet_frames_override is not None else int(quiet_seconds * fps)
    total_frames = quiet_frames + accel_frames
    head = np.zeros((total_frames, 263), dtype=np.float32)

    reference_frame_corrected = _height_corrected_reference(reference_frame, first_frame)

    for i in range(quiet_frames):
        frame = reference_frame_corrected.copy()
        frame[0:3] = 0.0
        frame[3] = first_frame[3]

        head[i] = frame

    for i in range(accel_frames):
        t = 1.0 - (i + 1) / accel_frames
        head[quiet_frames + i] = interpolate_frame_towards_reference(
            first_frame, reference_frame_corrected, t,
            lagging_foot=leading_foot, effective_max_lift=effective_max_lift
        )

    return head


def pad_to_uniform_length_symmetric(cut_augmented, reference_frame,
                                     target_total_length=UNIFORM_TARGET_LENGTH,
                                     noise_std=0.001):
    current_length = cut_augmented.shape[0]

    if current_length > target_total_length:
        return None
    if current_length == target_total_length:
        return cut_augmented

    extra_frames_needed = target_total_length - current_length
    pad_before = extra_frames_needed // 2
    pad_after = extra_frames_needed - pad_before

    first_boundary = cut_augmented[0]
    last_boundary = cut_augmented[-1]

    padding_before = np.tile(first_boundary, (pad_before, 1)).astype(np.float32)
    padding_before[:, 0:3] = 0.0

    padding_after = np.tile(last_boundary, (pad_after, 1)).astype(np.float32)
    padding_after[:, 0:3] = 0.0

    return np.concatenate([padding_before, cut_augmented, padding_after], axis=0)


def pad_to_uniform_length(cut_augmented, reference_frame,
                           target_total_length=UNIFORM_TARGET_LENGTH,
                           noise_std=0.001):
    current_length = cut_augmented.shape[0]

    if current_length > target_total_length:
        return None

    if current_length == target_total_length:
        return cut_augmented

    extra_frames_needed = target_total_length - current_length
    boundary_frame = cut_augmented[-1]
    extra_padding = np.tile(boundary_frame, (extra_frames_needed, 1)).astype(np.float32)
    extra_padding[:, 0:3] = 0.0

    return np.concatenate([cut_augmented, extra_padding], axis=0)