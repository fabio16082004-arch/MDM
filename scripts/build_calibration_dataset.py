#!/usr/bin/env python
"""
Punto 3 della metodologia: costruisce un dataset offline di esempi con
crop VARIABILE + SLERP + padding uniforme applicati, salvando
esplicitamente il metadato t_cut (l'indice esatto dell'ultimo frame
originale, prima della coda sintetica) per ciascun esempio.

Questo dataset serve ESCLUSIVAMENTE per calibrare l'algoritmo di
rilevamento del cutoff (Punto 4) - non sostituisce CustomDatasetLoader.

USO (dalla ROOT del progetto):
    python scripts/build_calibration_dataset.py \
        --step_index splits/step_index.json --n_per_class 150 \
        --output calibration_dataset.npz
"""
import argparse
import json
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loaders.slerp_utilities import build_slerp_tail, pad_to_uniform_length
from data_loaders.custom_dataloaders import sample_random_crop, min_frames_for_class


def get_reference_frame(donor_path, end_frame, quiet_cache, fallback_reference):
    quiet_frame_indices = quiet_cache.get(donor_path, [])
    if quiet_frame_indices:
        closest = min(quiet_frame_indices, key=lambda f: abs(f - end_frame))
        data = np.load(donor_path)
        return data[closest].astype(np.float32)
    return fallback_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--classes', default='1,2,3,4')
    parser.add_argument('--direction', default='forward')
    parser.add_argument('--n_per_class', type=int, default=150)
    parser.add_argument('--quiet_reference_path', default='assets/custom_augmentation/quiet_reference_pose.npy')
    parser.add_argument('--quiet_frame_cache', default='assets/custom_augmentation/quiet_frame_cache.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.step_index, encoding='utf-8') as f:
        step_index = json.load(f)

    fallback_reference = np.load(args.quiet_reference_path)[0].astype(np.float32)
    with open(args.quiet_frame_cache, encoding='utf-8') as f:
        quiet_cache = json.load(f)

    classes = args.classes.split(',')

    all_motions = []
    all_t_cuts = []
    all_classes = []
    all_donor_paths = []

    for L in classes:
        entries = step_index.get(L, [])
        if args.direction != 'any':
            entries = [e for e in entries if e.get('direction') == args.direction]

        n = min(args.n_per_class, len(entries))
        sampled = random.sample(entries, n)

        min_frames = min_frames_for_class(int(L))
        n_kept, n_discarded = 0, 0

        for entry in sampled:
            donor_path = entry['donor_path']
            try:
                data = np.load(donor_path)
            except Exception:
                continue

            rng = random.Random(hash((donor_path, entry['start_frame'], entry['end_frame'])))
            crop_start, crop_end = sample_random_crop(
                rng, entry['start_frame'], entry['end_frame'],
                entry['core_start'], entry['core_end'],
                min_frames=min_frames
            )

            cut_original = data[crop_start:crop_end].astype(np.float32)
            if cut_original.shape[0] < 2:
                continue

            # t_cut: l'indice (relativo alla sequenza finale) dell'ULTIMO
            # frame originale, prima che inizi la coda SLERP.
            t_cut = cut_original.shape[0]

            reference_frame = get_reference_frame(donor_path, crop_end, quiet_cache, fallback_reference)
            tail = build_slerp_tail(data, crop_end, reference_frame, class_L=int(L))
            cut_augmented = np.concatenate([cut_original, tail], axis=0)

            cut_padded = pad_to_uniform_length(cut_augmented, reference_frame)
            if cut_padded is None:
                n_discarded += 1
                continue

            all_motions.append(cut_padded)
            all_t_cuts.append(t_cut)
            all_classes.append(int(L))
            all_donor_paths.append(donor_path)
            n_kept += 1

        print(f"Classe {L}: {n_kept} tenuti, {n_discarded} scartati (outlier troppo lunghi)")

    motions_array = np.stack(all_motions, axis=0)
    t_cuts_array = np.array(all_t_cuts, dtype=np.int32)
    classes_array = np.array(all_classes, dtype=np.int32)

    np.savez(args.output,
             motions=motions_array,
             t_cuts=t_cuts_array,
             classes=classes_array,
             donor_paths=np.array(all_donor_paths))

    print(f"\n=== Fatto: {len(all_motions)} esempi salvati in '{args.output}' ===")
    print(f"Shape motions: {motions_array.shape}")
    print(f"t_cut - min: {t_cuts_array.min()}, max: {t_cuts_array.max()}, media: {t_cuts_array.mean():.1f}")


if __name__ == '__main__':
    main()