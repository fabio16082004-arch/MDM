# scripts/confusion_matrix.py — no exclusion count in image title
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from count_with_tolerance import (
    compute_foot_contact_from_joints,
    count_steps_with_calibrated_cutoff,
)
from count_without_cutoff import count_steps_in_generated_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_npy', required=True)
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--min_horizontal_m', type=float, default=0.15)
    parser.add_argument('--min_height_m', type=float, default=0.02)
    parser.add_argument('--min_straightness', type=float, default=0.3)
    parser.add_argument('--foot_detect_thres', type=float, default=0.002)
    parser.add_argument('--cutoff_eps', type=float, default=0.01)
    parser.add_argument('--cutoff_min_plateau', type=int, default=18)
    parser.add_argument('--cutoff_search_start', type=int, default=10)
    parser.add_argument('--metric', choices=['with_cutoff', 'no_cutoff'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='')
    parser.add_argument('--vmax', type=int, default=100)
    args = parser.parse_args()

    args.results_npy = os.path.abspath(args.results_npy)
    args.step_index = os.path.abspath(args.step_index)

    from data_loaders.get_data import get_dataset_loader
    loader = get_dataset_loader(
        name='custom_dataset', batch_size=1, num_frames=None,
        split=args.split, device='cpu', step_index_path=args.step_index,
    )
    dataset = loader.dataset
    expected_per_sample = [int(dataset.fixed_items[i]['L']) for i in range(len(dataset))]

    raw = np.load(args.results_npy, allow_pickle=True).item()
    motion = raw['motion']
    lengths = raw['lengths']
    num_samples = raw['num_samples']
    num_repetitions = raw['num_repetitions']

    if len(expected_per_sample) != num_samples:
        raise SystemExit(f"Mismatch: {len(expected_per_sample)} expected classes vs "
                         f"num_samples={num_samples} in results.npy")

    expected_classes = expected_per_sample * num_repetitions
    bs = motion.shape[0]

    classes = [1, 2, 3, 4]
    matrix = np.zeros((4, 4), dtype=int)
    n_excluded = 0
    excluded_by_class = {1: 0, 2: 0, 3: 0, 4: 0}

    print(f"Metric used: {args.metric}")

    for i in range(bs):
        joints_i = motion[i].transpose(2, 0, 1)
        contact_i = compute_foot_contact_from_joints(joints_i, args.foot_detect_thres)
        length_i = int(lengths[i])
        expected_L = expected_classes[i]

        if args.metric == 'with_cutoff':
            n_steps = count_steps_with_calibrated_cutoff(
                joints_i, contact_i, length_i,
                args.min_horizontal_m, args.min_height_m, args.min_straightness,
                cutoff_eps=args.cutoff_eps, cutoff_min_plateau=args.cutoff_min_plateau,
                cutoff_search_start=args.cutoff_search_start
            )
        else:
            n_steps, _, _ = count_steps_in_generated_video(
                joints_i, contact_i, length_i,
                args.min_horizontal_m, args.min_height_m, args.min_straightness
            )

        if n_steps == 0:
            n_excluded += 1
            excluded_by_class[expected_L] += 1
            continue

        detected_clipped = max(1, min(4, n_steps))
        matrix[expected_L - 1, detected_clipped - 1] += 1

    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0) * 100

    print(f"\nExcluded examples (count=0): {n_excluded} / {bs} ({100*n_excluded/bs:.1f}%)")
    print("Excluded per expected class:", excluded_by_class)

    print("\nConfusion matrix (absolute count, zero-count examples excluded):")
    print(matrix)
    print("\nConfusion matrix (row-wise percentage, over remaining examples):")
    print(np.round(matrix_pct, 1))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    norm = mcolors.Normalize(vmin=0, vmax=args.vmax)
    im = ax.imshow(matrix_pct, cmap='Blues', norm=norm)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"{c}" for c in classes])
    ax.set_yticklabels([f"{c}" for c in classes])
    ax.set_xlabel("Detected class")
    ax.set_ylabel("Requested class")
    if args.title:
        ax.set_title(args.title, fontsize=10)

    for i in range(4):
        for j in range(4):
            text_color = "white" if matrix_pct[i, j] > args.vmax * 0.6 else "black"
            ax.text(j, i, f"{matrix_pct[i, j]:.1f}%\n(n={matrix[i, j]})",
                    ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, label="% of requested class")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()