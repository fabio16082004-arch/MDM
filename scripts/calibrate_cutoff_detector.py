#!/usr/bin/env python
import argparse

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
))
from dataset_index import recover_joints

ROOT_JOINT_IDX = 0


def detect_cutoff(joints, eps, min_plateau_frames, search_start=0):
    T = joints.shape[0]
    root_xz = joints[:, ROOT_JOINT_IDX, [0, 2]]

    for start in range(search_start, T - min_plateau_frames):
        end = start + min_plateau_frames
        window = root_xz[start:end]
        speeds = np.linalg.norm(np.diff(window, axis=0), axis=1)
        avg_speed = speeds.mean() if len(speeds) > 0 else 0.0
        if avg_speed < eps:
            return start

    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_dataset', required=True)
    parser.add_argument('--eps_grid', default='0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05')
    parser.add_argument('--n_grid', default='5,6,7,8,9,10,12,15')
    args = parser.parse_args()

    data = np.load(args.calibration_dataset, allow_pickle=True)
    motions = data['motions']
    t_cuts_true = data['t_cuts']
    classes = data['classes']

    print(f"Precalcolo i giunti globali per {len(motions)} esempi...")
    all_joints = []
    for i in range(len(motions)):
        joints = recover_joints(motions[i])
        all_joints.append(joints)

    eps_values = [float(x) for x in args.eps_grid.split(',')]
    n_values = [int(x) for x in args.n_grid.split(',')]

    best_mae = float('inf')
    best_params = None
    results = []

    print(f"\nGrid search su {len(eps_values)} x {len(n_values)} = "
          f"{len(eps_values)*len(n_values)} combinazioni...\n")

    for eps in eps_values:
        for n_plateau in n_values:
            errors = []
            for i, joints in enumerate(all_joints):
                estimated = detect_cutoff(joints, eps, n_plateau)
                true_cut = t_cuts_true[i]
                errors.append(abs(estimated - true_cut))

            mae = np.mean(errors)
            median_ae = np.median(errors)
            results.append((eps, n_plateau, mae, median_ae))

            if mae < best_mae:
                best_mae = mae
                best_params = (eps, n_plateau)

    results.sort(key=lambda r: r[2])

    print("=== Top 10 combinazioni (ordinate per MAE) ===")
    print(f"{'eps':<8} {'N_plateau':<10} {'MAE (frame)':<14} {'Mediana AE':<12}")
    for eps, n_plateau, mae, median_ae in results[:10]:
        print(f"{eps:<8.3f} {n_plateau:<10} {mae:<14.2f} {median_ae:<12.2f}")

    print(f"\n=== Migliore combinazione ===")
    print(f"eps={best_params[0]}, N_plateau={best_params[1]}, MAE={best_mae:.2f} frame")

    print(f"\n=== Errore per classe (con i parametri migliori) ===")
    for L in sorted(set(classes)):
        mask = classes == L
        errors_L = []
        for i in np.where(mask)[0]:
            estimated = detect_cutoff(all_joints[i], best_params[0], best_params[1])
            errors_L.append(abs(estimated - t_cuts_true[i]))
        print(f"Classe {L}: MAE={np.mean(errors_L):.2f} frame  "
              f"(n={mask.sum()}, mediana={np.median(errors_L):.2f})")


if __name__ == '__main__':
    main()