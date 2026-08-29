#!/usr/bin/env python
"""
Punto 2 della metodologia: verifica che il rapporto hold_frames/total_frames
non sia troppo informativo sulla classe (numero di passi) da solo - cioe'
che il modello non possa "imbrogliare" contando i passi guardando quanto
dura proporzionalmente la coda di chiusura, invece del vero movimento.

USO (dalla ROOT del progetto):
    python scripts/verify_hold_ratio_correlation.py \
        --step_index splits/step_index.json --n_per_class 100 \
        --quiet_seconds 0.5
"""
import argparse
import json
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loaders.slerp_utilities import DECEL_FRAMES_PER_CLASS
from data_loaders.custom_dataloaders import sample_random_crop, min_frames_for_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--classes', default='1,2,3,4')
    parser.add_argument('--direction', default='forward')
    parser.add_argument('--n_per_class', type=int, default=100)
    parser.add_argument('--quiet_seconds', type=float, default=0.5)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.step_index, encoding='utf-8') as f:
        step_index = json.load(f)

    classes = args.classes.split(',')
    all_L = []
    all_ratios = []

    quiet_frames_fixed = int(args.quiet_seconds * args.fps)

    for L in classes:
        entries = step_index.get(L, [])
        if args.direction != 'any':
            entries = [e for e in entries if e.get('direction') == args.direction]

        n = min(args.n_per_class, len(entries))
        sampled = random.sample(entries, n)

        ratios_this_class = []
        orig_lens_this_class = []

        decel_frames = DECEL_FRAMES_PER_CLASS.get(int(L), 10)
        tail_len = decel_frames + quiet_frames_fixed

        for entry in sampled:
            donor_path = entry['donor_path']
            try:
                data = np.load(donor_path)
            except Exception:
                continue

            min_frames = min_frames_for_class(int(L))
            rng = random.Random(hash((donor_path, entry['start_frame'], entry['end_frame'])))
            crop_start, crop_end = sample_random_crop(
                rng, entry['start_frame'], entry['end_frame'],
                entry['core_start'], entry['core_end'],
                min_frames=min_frames
            )
            orig_len = crop_end - crop_start
            if orig_len < 2:
                continue

            total_len = orig_len + tail_len
            ratio = tail_len / total_len

            all_L.append(int(L))
            all_ratios.append(ratio)
            ratios_this_class.append(ratio)
            orig_lens_this_class.append(orig_len)

        print(f"Classe {L}: n={len(ratios_this_class)}  "
              f"orig_len medio={np.mean(orig_lens_this_class):.1f}  "
              f"tail_len={tail_len}  "
              f"ratio medio={np.mean(ratios_this_class):.3f}  "
              f"ratio std={np.std(ratios_this_class):.3f}")

    all_L = np.array(all_L)
    all_ratios = np.array(all_ratios)

    correlation = np.corrcoef(all_L, all_ratios)[0, 1]

    print(f"\n=== Risultato ===")
    print(f"Correlazione di Pearson (classe vs rapporto coda/totale): {correlation:.4f}")

    if abs(correlation) > 0.5:
        print("[ATTENZIONE] Correlazione FORTE - il rapporto coda/totale e' un segnale "
              "facilmente sfruttabile dal modello per indovinare la classe SENZA contare "
              "i veri passi. Considerare un padding tecnico per uniformare le lunghezze totali.")
    elif abs(correlation) > 0.25:
        print("[NOTA] Correlazione moderata - da monitorare, ma probabilmente non dominante.")
    else:
        print("[OK] Correlazione debole - il rapporto coda/totale non sembra un segnale "
              "dominante per distinguere le classi.")


if __name__ == '__main__':
    main()