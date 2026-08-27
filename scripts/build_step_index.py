#!/usr/bin/env python
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
))
from dataset_index import root_trajectory, recover_joints, FID_L, FID_R, CONTACT_SLICE


# ============================================================
# STADIO 1 - Rilevamento grezzo via foot-contact
# ============================================================

def find_candidate_step_windows(data):
    foot_contacts = (data[:, CONTACT_SLICE] > 0.5).astype(int)

    results = []
    pending_liftoff = {'left': None, 'right': None}

    last_left = foot_contacts[0][0] | foot_contacts[0][1]
    last_right = foot_contacts[0][2] | foot_contacts[0][3]

    for t in range(1, len(data)):
        step_left = foot_contacts[t][0] | foot_contacts[t][1]
        step_right = foot_contacts[t][2] | foot_contacts[t][3]

        if last_left == 1 and step_left == 0:
            pending_liftoff['left'] = t
        if last_left == 0 and step_left == 1 and pending_liftoff['left'] is not None:
            results.append({'foot': 'left', 'liftoff': pending_liftoff['left'], 'touchdown': t})
            pending_liftoff['left'] = None

        if last_right == 1 and step_right == 0:
            pending_liftoff['right'] = t
        if last_right == 0 and step_right == 1 and pending_liftoff['right'] is not None:
            results.append({'foot': 'right', 'liftoff': pending_liftoff['right'], 'touchdown': t})
            pending_liftoff['right'] = None

        last_left, last_right = step_left, step_right

    results.sort(key=lambda r: r['touchdown'])
    return results


# ============================================================
# STADIO 2 - Validazione geometrica
# ============================================================

def is_step_valid(joints, liftoff, touchdown, foot,
                   min_horizontal_m=0.15, min_height_m=0.02):
    """
    joints: posizioni GLOBALI (T, 22, 3), gia' calcolate una volta sola
            per l'intero donatore (recover_joints), non ricalcolate qui.
    Valido solo se ENTRAMBE le soglie sono superate (AND, non OR):
      - spostamento orizzontale xz della caviglia >= min_horizontal_m
      - altezza massima raggiunta dalla punta (relativa al minimo nella
        stessa finestra) >= min_height_m
    """
    ankle_idx = FID_L[0] if foot == 'left' else FID_R[0]   # 7 or 8
    toe_idx   = FID_L[1] if foot == 'left' else FID_R[1]   # 10 or 11

    ankle_xz_start = joints[liftoff, ankle_idx, [0, 2]]
    ankle_xz_end   = joints[touchdown, ankle_idx, [0, 2]]
    horizontal_distance = np.linalg.norm(ankle_xz_end - ankle_xz_start)

    toe_heights = [joints[t, toe_idx, 1] for t in range(liftoff, touchdown + 1)]
    max_vertical_distance = max(toe_heights) - min(toe_heights)

    return horizontal_distance >= min_horizontal_m and max_vertical_distance >= min_height_m


def get_validated_step_timeline(data, min_horizontal_m=0.15, min_height_m=0.02):
    """
    Applica find_candidate_step_windows + is_step_valid a un donatore
    intero, fondendo sinistro/destro in un'unica timeline ordinata per
    'touchdown', ciascuno marcato con 'valid': bool.
    """
    joints = recover_joints(data)   # calcolato UNA SOLA VOLTA per donatore
    footstep_candidates = find_candidate_step_windows(data)
    for candidate in footstep_candidates:
        candidate['valid'] = is_step_valid(
            joints=joints,
            liftoff=candidate['liftoff'],
            touchdown=candidate['touchdown'],
            foot=candidate['foot'],
            min_horizontal_m=min_horizontal_m,
            min_height_m=min_height_m,
        )
    return footstep_candidates


def group_into_blocks(step_timeline):
    """
    Spezza la timeline in blocchi MASSIMALI di passi validi consecutivi
    (nessun passo non valido in mezzo).
    """
    blocks = []
    arr = []
    for step in step_timeline:
        if step['valid']:
            arr.append(step)
        else:
            if arr:
                blocks.append(arr)
            arr = []
    if arr:
        blocks.append(arr)
    return blocks


def block_is_genuine_walking(data, block, min_straightness=0.3):
    start = block[0]['liftoff']
    end = block[-1]['touchdown']

    traj = root_trajectory(data[start:end + 1])
    net_displacement = np.linalg.norm(traj[-1] - traj[0])
    path_length = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    straightness = net_displacement / path_length if path_length > 1e-6 else 0.0

    return straightness >= min_straightness


# ============================================================
# STADIO 3 - Estrazione finestre sicure (zona massima) + nucleo
# ============================================================

def detect_direction(data, start, end):
    """
    Classifica la direzione dominante del movimento in [start:end),
    usando le velocita' LOCALI del root (canali 1:3 del vettore vec263).
    Convenzione HumanML3D: canale 2 (z locale) = avanti(+)/indietro(-),
    canale 1 (x locale) = laterale.

    Ritorna: 'forward' | 'backward' | 'lateral' | None (fermo)
    """
    local_vx = data[start:end, 1]
    local_vz = data[start:end, 2]
    net_x, net_z = local_vx.sum(), local_vz.sum()

    if abs(net_z) < 1e-3 and abs(net_x) < 1e-3:
        return None

    if abs(net_z) >= abs(net_x):
        return 'forward' if net_z > 0 else 'backward'
    else:
        return 'lateral'


def min_sec_for_category(category, min_sec_1, min_sec_2, min_sec_multi):
    """
    Risolve la durata minima (in secondi) da richiedere per una data
    categoria (numero di passi L).

    Priorita':
      - categoria 1  -> sempre min_sec_1
      - categoria 2  -> min_sec_2 se specificato (non None),
                        altrimenti fallback su min_sec_multi
      - categoria >=3 -> sempre min_sec_multi
    """
    if category == 1:
        return min_sec_1
    if category == 2:
        return min_sec_2 if min_sec_2 is not None else min_sec_multi
    return min_sec_multi


def enumerate_windows_from_block(donor_path, data, block, categories,
                                  min_sec_1=1.5, min_sec_2=None,
                                  min_sec_multi=2.0, fps=20):
    """
    Per un blocco di R passi validi consecutivi, per ogni L in categories
    con L <= R, enumera TUTTE le finestre (i, i+L-1) possibili.

    Per ciascuna, calcola:
      - il NUCLEO (core_start, core_end): i frame esatti in cui avviene
        fisicamente il movimento (dal liftoff del primo passo selezionato
        al touchdown dell'ultimo) - va SEMPRE incluso per intero in
        qualunque crop scelto poi a runtime.
      - la ZONA MASSIMA SICURA (start_frame, end_frame): quanto ci si puo'
        allargare attorno al nucleo senza mai toccare un passo adiacente
        NE' sconfinare fuori dal blocco stesso (territorio non validato).
        Ai bordi del blocco (i==0 o ultima finestra), il limite e' il
        blocco stesso (block_start / block_end+1), MAI l'intero donatore.

    Una finestra viene scartata se la zona sicura non e' abbastanza ampia
    da contenere almeno la durata minima richiesta per quella classe
    (risolta da min_sec_for_category) - il taglio finale esatto (durata
    e posizione) viene scelto solo a runtime, nel dataloader (temporal
    random cropping), non qui.

    Ritorna: lista di dict {'L', 'donor_path', 'start_frame', 'end_frame',
                            'core_start', 'core_end', 'direction'}
    """
    video_windows = []
    R = len(block)
    block_start = block[0]['liftoff']
    block_end = block[-1]['touchdown']

    for category in categories:
        min_sec = min_sec_for_category(category, min_sec_1, min_sec_2, min_sec_multi)
        target_frames = int(min_sec * fps)

        if R >= category:
            for i in range(0, R - category + 1):
                core_liftoff = block[i]['liftoff']
                core_touchdown = block[i + category - 1]['touchdown']


                max_safe_start = block[i - 1]['touchdown'] + 1 if i > 0 else block_start
                max_safe_end = block[i + category]['liftoff'] - 1 if (i + category) < R else block_end + 1

                if max_safe_end - max_safe_start < target_frames:
                    continue

                direction = detect_direction(data, core_liftoff, core_touchdown)

                video_windows.append({
                    'L': category,
                    'donor_path': donor_path,
                    'start_frame': int(max_safe_start),
                    'end_frame': int(max_safe_end),
                    'core_start': int(core_liftoff),
                    'core_end': int(core_touchdown),
                    'direction': direction,
                })

    return video_windows


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--donors_list', required=True)
    parser.add_argument('--output_index', required=True)
    parser.add_argument('--categories', default='1,2,3,4')
    parser.add_argument('--min_sec_1', type=float, default=1.5,
                         help="Durata minima in sec per la classe 1 passo (default 1.5)")
    parser.add_argument('--min_sec_2', type=float, default=None,
                         help="Durata minima in sec per la classe 2 passi. "
                              "Se omesso, usa lo stesso valore di --min_sec_multi "
                              "(comportamento identico allo script originale).")
    parser.add_argument('--min_sec_multi', type=float, default=2.0,
                         help="Durata minima in sec per le classi 3+ passi "
                              "(e per la classe 2 se --min_sec_2 non e' specificato). Default 2.0")
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--min_horizontal_m', type=float, default=0.15)
    parser.add_argument('--min_height_m', type=float, default=0.02)
    parser.add_argument('--min_straightness', type=float, default=0.3)
    args = parser.parse_args()

    categories = [int(c) for c in args.categories.split(',')]

    with open(args.donors_list, encoding='utf-8') as f:
        donor_paths = [line.strip() for line in f if line.strip()]

    print(f"Donatori da processare: {len(donor_paths)}")
    print(f"Soglie durata minima: classe1={args.min_sec_1}s | "
          f"classe2={args.min_sec_2 if args.min_sec_2 is not None else args.min_sec_multi}s "
          f"({'esplicita' if args.min_sec_2 is not None else 'fallback su min_sec_multi'}) | "
          f"classi3+={args.min_sec_multi}s")

    index_by_class = {L: [] for L in categories}
    n_ok, n_skipped = 0, 0
    n_blocks_total, n_blocks_discarded_walking = 0, 0

    for i, donor_path in enumerate(donor_paths, 1):
        if i % 25 == 0 or i == len(donor_paths):
            print(f"  [{i}/{len(donor_paths)}] elaborati...", flush=True)

        try:
            data = np.load(donor_path)
            if data.ndim != 2 or data.shape[-1] < 263:
                n_skipped += 1
                continue

            timeline = get_validated_step_timeline(data, args.min_horizontal_m, args.min_height_m)
            blocks = group_into_blocks(timeline)
            n_blocks_total += len(blocks)

            genuine_blocks = []
            for block in blocks:
                if block_is_genuine_walking(data, block, args.min_straightness):
                    genuine_blocks.append(block)
                else:
                    n_blocks_discarded_walking += 1

            for block in genuine_blocks:
                windows = enumerate_windows_from_block(
                    donor_path=donor_path,
                    data=data,
                    block=block,
                    categories=categories,
                    min_sec_1=args.min_sec_1,
                    min_sec_2=args.min_sec_2,
                    min_sec_multi=args.min_sec_multi,
                    fps=args.fps
                )
                for w in windows:
                    index_by_class[w['L']].append(w)

            n_ok += 1

        except Exception as e:
            print(f"[ERRORE] {donor_path}: {e}")
            n_skipped += 1
            continue

    with open(args.output_index, 'w', encoding='utf-8') as f:
        json.dump(index_by_class, f)

    print(f"\n=== Riepilogo ===")
    print(f"Successo: {n_ok} | Errori: {n_skipped}")
    print(f"Blocchi totali: {n_blocks_total} | scartati per 'marcia sul posto' "
          f"(rettilineita' < {args.min_straightness}): {n_blocks_discarded_walking}")
    for L in categories:
        print(f"  {L}_step: {len(index_by_class[L])} finestre massime sicure trovate")
    print(f"\nIndice salvato in: {args.output_index}")


if __name__ == '__main__':
    main()