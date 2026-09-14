#!/usr/bin/env python
"""
Script di rendering comparativo, da lanciare al termine di una
generazione: per ciascun esempio nel results.npy, produce un unico
video MP4 con affiancati orizzontalmente:
  1) il video GENERATO dal checkpoint corrente
  2) la GROUND TRUTH corrispondente
  3) (se --baseline_results_npy e' fornito) il video dello stesso
     esempio al checkpoint 0k

Nel titolo del pannello "generato" viene inserito il numero di passi
rilevato dalla metrica corrente (count_steps_with_calibrated_cutoff).

USO (dalla ROOT del progetto):
    python scripts/render_comparison.py \
        --results_npy save/my_run/samples_test_40000_split-test/results.npy \
        --step_index splits/step_index.json --split test \
        --outdir scripts/render_comparison_40k \
        [--baseline_results_npy save/my_run/samples_test_0_split-test/results.npy] \
        [--n_examples 20]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_guofeats import guofeats_to_joints
from render_guofeats_multi import raw_joints_to_render_joints

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loaders.custom_dataloaders import FixedEvalDatasetLoader
from data_loaders.slerp_utilities import build_slerp_tail, build_slerp_head, UNIFORM_TARGET_LENGTH
from count_with_tolerance import compute_foot_contact_from_joints, count_steps_with_calibrated_cutoff


COLORS = ("black", "magenta", "red", "green", "blue")
KINEMATIC_TREE = [
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
]


def init_axis(ax, title, radius=1.5):
    ax.view_init(elev=20.0, azim=-60)
    fact = 2
    ax.set_xlim3d([-radius / fact, radius / fact])
    ax.set_ylim3d([-radius / fact, radius / fact])
    ax.set_zlim3d([0, radius])
    ax.set_aspect("auto")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()
    ax.grid(False)
    ax.set_title(title, loc="center", wrap=True, fontsize=9)
    return ax


def plot_floor(ax, minx, maxx, miny, maxy, minz):
    verts = [[minx, miny, minz], [minx, maxy, minz], [maxx, maxy, minz], [maxx, miny, minz]]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.add_collection3d(xz_plane)

    radius = max((maxx - minx), (maxy - miny))
    minx_all, maxx_all = (maxx + minx) / 2 - radius, (maxx + minx) / 2 + radius
    miny_all, maxy_all = (maxy + miny) / 2 - radius, (maxy + miny) / 2 + radius
    verts = [[minx_all, miny_all, minz], [minx_all, maxy_all, minz],
             [maxx_all, maxy_all, minz], [maxx_all, miny_all, minz]]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)


def update_camera(ax, root, radius=1.5):
    fact = 2
    ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
    ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])


def prepare_panel(ax, joints, title):
    init_axis(ax, title)
    joints = joints.copy()
    x, y, z = 0, 1, 2

    trajectory = joints[:, 0, [x, y]]
    avg_seg = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3 \
        if len(trajectory) > 1 else 1e-3
    draw_offset = int(25 / avg_seg)
    (spline_line,) = ax.plot(*trajectory.T, zorder=10, color="white")

    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax, minx, maxx, miny, maxy, 0)

    height_offset = np.min(joints[:, :, z])
    joints[:, :, z] -= height_offset

    lines = []
    return {'joints': joints, 'trajectory': trajectory, 'draw_offset': draw_offset,
            'spline_line': spline_line, 'lines': lines, 'initialized': False}


def update_panel(ax, state, frame):
    joints = state['joints']
    frame = min(frame, joints.shape[0] - 1)
    skeleton = joints[frame]
    x, y, z = 0, 1, 2

    root = skeleton[0]
    update_camera(ax, root)

    for index, (chain, color) in enumerate(zip(reversed(KINEMATIC_TREE), reversed(COLORS))):
        if not state['initialized']:
            state['lines'].append(ax.plot(
                skeleton[chain, x], skeleton[chain, y], skeleton[chain, z],
                linewidth=6.0, color=color, zorder=20,
                path_effects=[pe.SimpleLineShadow(), pe.Normal()],
            ))
        else:
            state['lines'][index][0].set_xdata(skeleton[chain, x])
            state['lines'][index][0].set_ydata(skeleton[chain, y])
            state['lines'][index][0].set_3d_properties(skeleton[chain, z])

    trajectory = state['trajectory']
    draw_offset = state['draw_offset']
    left = max(frame - draw_offset, 0)
    right = min(frame + draw_offset, trajectory.shape[0])
    state['spline_line'].set_xdata(trajectory[left:right, 0])
    state['spline_line'].set_ydata(trajectory[left:right, 1])
    state['spline_line'].set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
    state['initialized'] = True


def render_side_by_side(panels, output, fps=20.0):
    n = len(panels)
    fig = plt.figure(figsize=(4 * n, 4))
    axes = [fig.add_subplot(1, n, i + 1, projection='3d') for i in range(n)]

    states = []
    for ax, p in zip(axes, panels):
        states.append(prepare_panel(ax, p['joints'], p['title']))

    max_frames = max(p['joints'].shape[0] for p in panels)

    def update(frame):
        for ax, state in zip(axes, states):
            update_panel(ax, state, frame)

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=max_frames, interval=1000 / fps, repeat=False)

    try:
        anim.save(output, fps=fps)
        print(f"Salvato: {output}")
    except Exception as e:
        from matplotlib.animation import PillowWriter
        gif_output = os.path.splitext(output)[0] + ".gif"
        print(f"Salvataggio MP4 fallito ({e}); provo con GIF...")
        anim.save(gif_output, writer=PillowWriter(fps=int(round(fps))))
        print(f"Salvato: {gif_output}")

    plt.close(fig)


def load_generated_joints(results_npy_path, idx):
    """
    Convenzione ORIGINALE (X, Y_up, Z_depth) - IDENTICA a quella usata
    da count_with_tolerance.py. Usata per il CONTEGGIO, cosi' il numero
    di passi rilevato qui combacia esattamente con quello del file
    principale (foot-contact dai giunti + validazione geometrica +
    troncamento calibrato).
    """
    raw = np.load(results_npy_path, allow_pickle=True).item()
    motion = raw['motion']
    lengths = raw['lengths']
    joints = motion[idx].transpose(2, 0, 1)
    length = int(lengths[idx])
    return joints[:length], length


def load_generated_joints_for_rendering(results_npy_path, idx):
    """
    Come load_generated_joints, ma con gli assi riordinati (X, -Z_depth,
    Y_up) tramite raw_joints_to_render_joints - necessario SOLO per il
    rendering visivo (altrimenti lo scheletro appare sdraiato invece
    che in piedi). MAI usare questa versione per il conteggio.
    """
    joints, length = load_generated_joints(results_npy_path, idx)
    joints_for_render = raw_joints_to_render_joints(joints)
    return joints_for_render, length


def load_ground_truth_joints(fixed_loader, idx):
    # VERSIONE ATTIVA: crop grezzo, nessuna coda/testa SLERP - coerente
    # con la configurazione v38 (senza SLERP).
    item = fixed_loader.fixed_items[idx]
    data = np.load(item['donor_path'])

    cut = data[item['start_frame']:item['end_frame']]
    cut_padded = cut

    joints = guofeats_to_joints(torch.from_numpy(cut_padded.astype(np.float32))).numpy()
    return joints, item['caption'], item['L']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_npy', required=True)
    parser.add_argument('--baseline_results_npy', default=None)
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n_examples', type=int, default=None)
    parser.add_argument('--min_horizontal_m', type=float, default=0.15)
    parser.add_argument('--min_height_m', type=float, default=0.02)
    parser.add_argument('--min_straightness', type=float, default=0.3)
    parser.add_argument('--foot_detect_thres', type=float, default=0.002)
    parser.add_argument('--cutoff_eps', type=float, default=0.01)
    parser.add_argument('--cutoff_min_plateau', type=int, default=15)
    parser.add_argument('--fps', type=float, default=20.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.step_index, encoding='utf-8') as f:
        step_index = json.load(f)

    from captions import build_hand_written_captions
    hand_written_captions = build_hand_written_captions()
    from data_loaders.get_data import CUSTOM_EVAL_PER_CLASS_COUNT

    donors_split_path = f"splits/{args.split}.txt"
    fixed_loader = FixedEvalDatasetLoader(
        step_index, hand_written_captions, CUSTOM_EVAL_PER_CLASS_COUNT,
        donors_split_path=donors_split_path,
    )

    raw_gen = np.load(args.results_npy, allow_pickle=True).item()
    n_total = raw_gen['motion'].shape[0]
    n_examples = args.n_examples or n_total

    print(f"Totale esempi disponibili: {n_total}, ne renderizzo "
          f"{min(n_examples, n_total, len(fixed_loader.fixed_items))}")

    for idx in range(min(n_examples, n_total, len(fixed_loader.fixed_items))):
        # Conteggio: SEMPRE con la convenzione originale (identica a
        # count_with_tolerance.py), MAI con quella riordinata per il render.
        gen_joints_count, gen_length = load_generated_joints(args.results_npy, idx)
        contact_i = compute_foot_contact_from_joints(gen_joints_count, args.foot_detect_thres)
        n_steps = count_steps_with_calibrated_cutoff(
            gen_joints_count, contact_i, gen_length,
            args.min_horizontal_m, args.min_height_m, args.min_straightness,
            cutoff_eps=args.cutoff_eps, cutoff_min_plateau=args.cutoff_min_plateau,
        )

        # Rendering: versione con assi riordinati, SOLO per la visualizzazione.
        gen_joints_render, _ = load_generated_joints_for_rendering(args.results_npy, idx)

        caption_gen = raw_gen['text'][idx] if idx < len(raw_gen['text']) else ""
        gen_title = f"GENERATO\n{caption_gen}\npassi rilevati: {n_steps}"

        panels = [{'joints': gen_joints_render, 'title': gen_title}]

        gt_joints, gt_caption, gt_L = load_ground_truth_joints(fixed_loader, idx)
        panels.append({'joints': gt_joints, 'title': f"GROUND TRUTH\n{gt_caption}"})

        if args.baseline_results_npy and os.path.exists(args.baseline_results_npy):
            baseline_raw = np.load(args.baseline_results_npy, allow_pickle=True).item()
            if idx < baseline_raw['motion'].shape[0]:
                baseline_joints, _ = load_generated_joints_for_rendering(args.baseline_results_npy, idx)
                panels.append({'joints': baseline_joints, 'title': "CHECKPOINT 0k"})

        out_path = os.path.join(args.outdir, f"comparison_{idx:03d}.mp4")
        print(f"[{idx+1}] {out_path} — passi rilevati: {n_steps}, classe attesa: {gt_L}")
        render_side_by_side(panels, out_path, fps=args.fps)


if __name__ == '__main__':
    main()