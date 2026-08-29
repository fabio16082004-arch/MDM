#!/usr/bin/env python
"""
Precalcola, per ogni donatore, TUTTE le finestre di quiete genuina
trovate in qualunque punto del file (non solo la prima) - necessario
perche' un file puo' attraversare contesti fisici diversi (es. terra
-> rampa -> pianerottolo rialzato), quindi una singola pausa "qualunque"
potrebbe essere a una quota incoerente con lo specifico crop che si
sta aumentando. Salvando tutte le pause, in fase di utilizzo si potra'
scegliere quella temporalmente piu' vicina al crop specifico.

Formato della cache (JSON):
    {
        "videos/1_step/000100.npy": [234, 890, ...],
        "videos/4_step/007309.npy": [],
        ...
    }

USO (dalla ROOT del progetto):
    python scripts/build_quiet_frame_cache.py \
        --step_index splits/step_index.json --classes 1,2,3,4 \
        --output assets/custom_augmentation/quiet_frame_cache.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
))
from dataset_index import root_trajectory, CONTACT_SLICE


def count_liftoff_events(contacts_slice):
    thresh = (contacts_slice > 0.5).astype(int)
    n_liftoffs = 0
    for col in range(thresh.shape[1]):
        transitions = np.diff(thresh[:, col])
        n_liftoffs += int(np.sum(transitions == -1))
    return n_liftoffs


def find_all_quiet_windows(data, min_pause_frames=10, max_root_speed_m=0.01,
                            window_stride=5, min_gap_between=20):
    """
    Scansiona l'intero file, ritorna la lista dei frame centrali di
    TUTTE le finestre di quiete trovate.
    """
    n_frames = len(data)
    traj_full = root_trajectory(data)
    contacts_full = data[:, CONTACT_SLICE]

    found = []
    last_found_center = -min_gap_between - 1

    for start in range(0, n_frames - min_pause_frames, window_stride):
        end = start + min_pause_frames
        center = (start + end) // 2

        if center - last_found_center < min_gap_between:
            continue

        traj_window = traj_full[start:end]
        speeds = np.linalg.norm(np.diff(traj_window, axis=0), axis=1)
        avg_speed = speeds.mean() if len(speeds) > 0 else 0.0

        if avg_speed >= max_root_speed_m:
            continue

        contacts_window = contacts_full[start:end]
        if count_liftoff_events(contacts_window) > 0:
            continue

        found.append(center)
        last_found_center = center

    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--classes', default='1,2,3,4')
    parser.add_argument('--direction', default='forward')
    parser.add_argument('--min_pause_frames', type=int, default=10)
    parser.add_argument('--max_root_speed_m', type=float, default=0.01)
    parser.add_argument('--min_gap_between', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.step_index, encoding='utf-8') as f:
        step_index = json.load(f)

    classes = args.classes.split(',')

    all_donors = set()
    for L in classes:
        entries = step_index.get(L, [])
        if args.direction != 'any':
            entries = [e for e in entries if e.get('direction') == args.direction]
        for e in entries:
            all_donors.add(e['donor_path'])

    print(f"Donatori unici da processare: {len(all_donors)}")

    cache = {}
    n_with_pauses, n_without = 0, 0
    total_pauses = 0

    for i, donor_path in enumerate(sorted(all_donors)):
        if (i + 1) % 25 == 0 or (i + 1) == len(all_donors):
            print(f"  [{i+1}/{len(all_donors)}] processati...", flush=True)

        try:
            data = np.load(donor_path)
        except Exception as e:
            print(f"[ERRORE] {donor_path}: {e}")
            cache[donor_path] = []
            n_without += 1
            continue

        quiet_frames = find_all_quiet_windows(
            data, args.min_pause_frames, args.max_root_speed_m,
            min_gap_between=args.min_gap_between
        )

        cache[donor_path] = quiet_frames
        total_pauses += len(quiet_frames)
        if quiet_frames:
            n_with_pauses += 1
        else:
            n_without += 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

    print(f"\n=== Riepilogo ===")
    print(f"Donatori con almeno una pausa: {n_with_pauses}/{len(all_donors)} "
          f"({100*n_with_pauses/len(all_donors):.1f}%)")
    print(f"Donatori senza alcuna pausa: {n_without}/{len(all_donors)}")
    print(f"Totale pause trovate (tutte insieme): {total_pauses}")
    print(f"Media pause per donatore (tra quelli con almeno una): "
          f"{total_pauses/max(n_with_pauses,1):.1f}")
    print(f"\nCache salvata in: {args.output}")


if __name__ == '__main__':
    main()