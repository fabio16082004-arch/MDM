#!/usr/bin/env python
"""
Applica sia il taglio (crop VARIABILE, tramite sample_random_crop) sia
l'augmentazione SLERP (decelerazione con arco di sollevamento + quiete)
a entries di step_index.json, esportando il risultato nel formato
compatibile con tools/dataset_viewer.py per ispezione visiva.

USO (dalla ROOT del progetto):
    python scripts/export_slerp_for_viewer.py \
        --step_index splits/step_index.json --classes 1,2,3,4 \
        --n_per_class 7 --quiet_seconds 0.5 \
        --outdir dataset_slerp_prototype_v2 \
        --quiet_reference_path assets/custom_augmentation/quiet_reference_pose.npy \
        --quiet_frame_cache assets/custom_augmentation/quiet_frame_cache.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_guofeats import guofeats_to_joints, render_animation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loaders.slerp_utilities import build_slerp_tail
from data_loaders.custom_dataloaders import sample_random_crop, min_frames_for_class


def get_reference_frame(donor_path, end_frame, quiet_cache, fallback_reference):
    quiet_frame_indices = quiet_cache.get(donor_path, [])
    if quiet_frame_indices:
        closest = min(quiet_frame_indices, key=lambda f: abs(f - end_frame))
        data = np.load(donor_path)
        return data[closest].astype(np.float32), f'per_donatore(frame={closest})'
    return fallback_reference, 'fallback_universale'


def load_donor_splits(splits_dir):
    donor_to_split = {}
    for sp in ('train', 'val', 'test'):
        p = os.path.join(splits_dir, f'{sp}.txt')
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    donor_to_split[os.path.normpath(line)] = sp
    return donor_to_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--classes', default='1,2,3,4')
    parser.add_argument('--direction', default='forward')
    parser.add_argument('--n_per_class', type=int, default=None)
    parser.add_argument('--crop_seed_offset', type=int, default=0)
    parser.add_argument('--quiet_reference_path', default='assets/custom_augmentation/quiet_reference_pose.npy')
    parser.add_argument('--quiet_frame_cache', default='assets/custom_augmentation/quiet_frame_cache.json')
    parser.add_argument('--quiet_seconds', type=float, default=0.5,
                         help="Durata della fase di quiete vera (default 0.5s).")
    parser.add_argument('--splits_dir', default='splits')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fps', type=float, default=20.0)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.step_index, encoding='utf-8') as f:
        step_index = json.load(f)

    fallback_reference = np.load(args.quiet_reference_path)[0].astype(np.float32)
    with open(args.quiet_frame_cache, encoding='utf-8') as f:
        quiet_cache = json.load(f)

    donor_to_split = load_donor_splits(args.splits_dir)

    classes = args.classes.split(',')
    total_exported = 0
    split_paths = {'train': [], 'val': [], 'test': []}

    for L in classes:
        entries = step_index.get(L, [])
        if args.direction != 'any':
            entries = [e for e in entries if e.get('direction') == args.direction]

        if args.n_per_class is not None:
            n = min(args.n_per_class, len(entries))
            entries = random.sample(entries, n)

        class_dir_name = f"{L}_step"
        class_dir = os.path.join(args.outdir, class_dir_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"\n=== Classe {L}: esporto {len(entries)} entries (crop + SLERP) ===")

        for i, entry in enumerate(entries):
            donor_path = entry['donor_path']

            try:
                data = np.load(donor_path)
            except Exception as e:
                print(f"  [ERRORE] {donor_path}: {e}")
                continue

            min_frames = min_frames_for_class(int(L))
            rng = random.Random(hash((donor_path, entry['start_frame'],
                                       entry['end_frame'])) + args.crop_seed_offset)
            crop_start, crop_end = sample_random_crop(
                rng, entry['start_frame'], entry['end_frame'],
                entry['core_start'], entry['core_end'],
                min_frames=min_frames
            )

            cut_original = data[crop_start:crop_end].astype(np.float32)
            if cut_original.shape[0] < 2:
                print(f"  [SALTATO] crop troppo corto: {donor_path}")
                continue

            reference_frame, ref_source = get_reference_frame(donor_path, crop_end, quiet_cache, fallback_reference)
            tail = build_slerp_tail(
                data, crop_end, reference_frame,
                class_L=int(L),
                quiet_seconds=args.quiet_seconds,
                fps=args.fps,
            )
            cut_augmented = np.concatenate([cut_original, tail], axis=0)

            donor_name = os.path.splitext(os.path.basename(donor_path))[0]
            base_name = f"{donor_name}_{crop_start}_{crop_end}_slerp"

            npy_path = os.path.join(class_dir, base_name + '.npy')
            txt_path = os.path.join(class_dir, base_name + '.txt')
            mp4_path = os.path.join(class_dir, base_name + '.mp4')

            np.save(npy_path, cut_augmented)

            caption = (f"L={L} step(s) + SLERP closing (quiet={args.quiet_seconds}s), "
                       f"donor={donor_name}, riferimento={ref_source}")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            split = donor_to_split.get(os.path.normpath(donor_path))
            if split:
                split_paths[split].append(npy_path)

            joints = guofeats_to_joints(torch.from_numpy(cut_augmented)).numpy()
            render_animation(joints, mp4_path,
                              title=f"{base_name} ({cut_augmented.shape[0]}f, "
                                    f"orig={cut_original.shape[0]}f)",
                              fps=args.fps)

            total_exported += 1
            print(f"  [{i+1}/{len(entries)}] {base_name}  "
                  f"orig={cut_original.shape[0]}f  tail={tail.shape[0]}f  "
                  f"totale={cut_augmented.shape[0]}f  riferimento={ref_source}")

    for sp, paths in split_paths.items():
        if not paths:
            continue
        out_path = os.path.join(args.outdir, f'{sp}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            for p in paths:
                f.write(p + '\n')

    print(f"\n=== Fatto: {total_exported} crop (con SLERP) esportati in '{args.outdir}/' ===")
    print(f"Visualizza con:\n  python tools/dataset_viewer.py {args.outdir}")


if __name__ == '__main__':
    main()