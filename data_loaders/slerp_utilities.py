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

def build_slerp_head(data, start_frame, reference_frame, class_L=None,
                      accel_frames=None, quiet_seconds=0.5, fps=20,
                      noise_std=0.001, use_foot_arc=True, max_lift=0.08,
                      lift_ratio=0.35, lookback=10):
    first_frame = data[start_frame].astype(np.float32)

    if accel_frames is None:
        accel_frames = DECEL_FRAMES_FIXED

    # FIX: il lookback per identify_lagging_foot guarda AVANTI da
    # start_frame (start_frame + lookback), non richiede frame precedenti
    # - ma limitiamo comunque lookback alla lunghezza disponibile nel file.
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

    quiet_frames = int(quiet_seconds * fps)
    total_frames = quiet_frames + accel_frames
    head = np.zeros((total_frames, 263), dtype=np.float32)

    for i in range(quiet_frames):
        frame = reference_frame.copy()
        frame[0:3] = 0.0
        frame[3] = first_frame[3]
        head[i] = frame

    for i in range(accel_frames):
        t = 1.0 - (i + 1) / accel_frames
        head[quiet_frames + i] = interpolate_frame_towards_reference(
            first_frame, reference_frame, t,
            lagging_foot=leading_foot, effective_max_lift=effective_max_lift
        )

    return head


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
                      noise_std=0.001, use_foot_arc=True, max_lift=0.08,
                      lift_ratio=0.35, lookback=10):
    """
    Genera la coda di transizione completa (decelerazione + quiete).

    decel_frames: se None, usa DECEL_FRAMES_FIXED (valore unico,
        indipendente dalla classe). class_L mantenuto per compatibilita'
        di interfaccia, ma non influenza piu' la durata.
    quiet_seconds: default 0.5s, coerente con la durata fisiologica
        tipica di una fase di doppio appoggio (0.4-0.6s).
    """
    last_frame = data[end_frame - 1].astype(np.float32)

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

    quiet_frames = int(quiet_seconds * fps)
    total_frames = decel_frames + quiet_frames
    tail = np.zeros((total_frames, 263), dtype=np.float32)

    for i in range(decel_frames):
        t = (i + 1) / decel_frames
        tail[i] = interpolate_frame_towards_reference(
            last_frame, reference_frame, t,
            lagging_foot=lagging_foot, effective_max_lift=effective_max_lift
        )

    last_decel_frame = tail[decel_frames - 1] if decel_frames > 0 else last_frame
    for i in range(quiet_frames):
        frame = reference_frame.copy()
        frame[0:3] = 0.0
        frame[3] = last_decel_frame[3]

        noise = np.random.normal(0, noise_std, size=frame.shape).astype(frame.dtype)
        noise[0:3] = 0.0      # velocita' (root)
        noise[3] = 0.0        # altezza assoluta (root)
        noise[4:67] = 0.0     # posizioni RIC dei giunti - NESSUN rumore, evita lo slittamento
        noise[193:259] = 0.0  # velocita' dei giunti
        noise[259:263] = 0.0  # foot-contact (binario, il rumore lo renderebbe incoerente)

        tail[decel_frames + i] = frame + noise

    return tail


def pad_to_uniform_length(cut_augmented, reference_frame,
                           target_total_length=UNIFORM_TARGET_LENGTH,
                           noise_std=0.001):
    current_length = cut_augmented.shape[0]

    if current_length > target_total_length:
        return None

    if current_length == target_total_length:
        return cut_augmented

    extra_frames_needed = target_total_length - current_length
    extra_padding = np.tile(reference_frame, (extra_frames_needed, 1)).astype(np.float32)
    extra_padding[:, 0:3] = 0.0

    noise = np.random.normal(0, noise_std, size=extra_padding.shape).astype(extra_padding.dtype)
    noise[:, 0:3] = 0.0      # velocita' (root)
    noise[:, 3] = 0.0        # altezza assoluta (root)
    noise[:, 4:67] = 0.0     # posizioni RIC - NESSUN rumore (stesso fix anti-slittamento)
    noise[:, 193:259] = 0.0  # velocita' dei giunti
    noise[:, 259:263] = 0.0  # foot-contact
    extra_padding = extra_padding + noise

    return np.concatenate([cut_augmented, extra_padding], axis=0)



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

    padding_before = np.tile(reference_frame, (pad_before, 1)).astype(np.float32)
    padding_before[:, 0:3] = 0.0

    padding_after = np.tile(reference_frame, (pad_after, 1)).astype(np.float32)
    padding_after[:, 0:3] = 0.0

    return np.concatenate([padding_before, cut_augmented, padding_after], axis=0)