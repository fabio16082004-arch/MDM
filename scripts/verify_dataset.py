import numpy as np
import json
import sys

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tools')
from dataset_index import recover_joints, CONTACT_SLICE
from build_step_index import is_step_valid


def find_candidate_step_windows_with_edge_fix(foot_contact):
    """
    Come find_candidate_step_windows_from_contact, con il fix del
    liftoff implicito: se un piede e' gia' sollevato al frame 0, si
    assume un liftoff avvenuto "prima dell'inizio osservabile".
    """
    contacts = (foot_contact > 0.5).astype(int)

    results = []
    pending_liftoff = {'left': None, 'right': None}
    is_edge_case = {'left': False, 'right': False}

    last_left = contacts[0][0] | contacts[0][1]
    last_right = contacts[0][2] | contacts[0][3]

    if last_left == 0:
        pending_liftoff['left'] = 0
        is_edge_case['left'] = True
    if last_right == 0:
        pending_liftoff['right'] = 0
        is_edge_case['right'] = True

    for t in range(1, len(contacts)):
        step_left = contacts[t][0] | contacts[t][1]
        step_right = contacts[t][2] | contacts[t][3]

        if last_left == 1 and step_left == 0:
            pending_liftoff['left'] = t
            is_edge_case['left'] = False
        if last_left == 0 and step_left == 1 and pending_liftoff['left'] is not None:
            results.append({
                'foot': 'left', 'liftoff': pending_liftoff['left'], 'touchdown': t,
                'is_edge_case': is_edge_case['left']
            })
            pending_liftoff['left'] = None
            is_edge_case['left'] = False

        if last_right == 1 and step_right == 0:
            pending_liftoff['right'] = t
            is_edge_case['right'] = False
        if last_right == 0 and step_right == 1 and pending_liftoff['right'] is not None:
            results.append({
                'foot': 'right', 'liftoff': pending_liftoff['right'], 'touchdown': t,
                'is_edge_case': is_edge_case['right']
            })
            pending_liftoff['right'] = None
            is_edge_case['right'] = False

        last_left, last_right = step_left, step_right

    results.sort(key=lambda r: r['touchdown'])
    return results


def count_valid_steps_with_edge_fix(joints, foot_contact, min_horizontal_m=0.15, min_height_m=0.02):
    """
    Applica il fix del liftoff implicito e valida ciascun candidato -
    per i casi di bordo, salta il controllo di distanza orizzontale
    (sottostimata per costruzione), verificando solo l'altezza.
    """
    candidates = find_candidate_step_windows_with_edge_fix(foot_contact)
    n_valid = 0
    for c in candidates:
        if c.get('is_edge_case', False):
            valid = is_step_valid(
                joints=joints, liftoff=c['liftoff'], touchdown=c['touchdown'],
                foot=c['foot'], min_horizontal_m=0.0, min_height_m=min_height_m,
            )
        else:
            valid = is_step_valid(
                joints=joints, liftoff=c['liftoff'], touchdown=c['touchdown'],
                foot=c['foot'], min_horizontal_m=min_horizontal_m, min_height_m=min_height_m,
            )
        if valid:
            n_valid += 1
    return n_valid, candidates


def count_steps_fixed_boundary(data, start_frame, end_frame,
                                min_horizontal_m=0.15, min_height_m=0.02):
    """Verifica sulla ground truth: guarda oltre il crop per il contesto completo."""
    from count_generated_steps import find_candidate_step_windows_from_contact
    contact_full = data[:, CONTACT_SLICE]
    candidates_full = find_candidate_step_windows_from_contact(contact_full)
    candidates_in_crop = [
        c for c in candidates_full
        if start_frame <= c['liftoff'] < end_frame
    ]
    joints_full = recover_joints(data)
    n_valid = 0
    for c in candidates_in_crop:
        valid = is_step_valid(
            joints=joints_full, liftoff=c['liftoff'], touchdown=c['touchdown'],
            foot=c['foot'], min_horizontal_m=min_horizontal_m, min_height_m=min_height_m
        )
        if valid:
            n_valid += 1
    return n_valid, candidates_in_crop


print("=" * 60)
print("VERIFICA 1: Ground truth (fix guarda-oltre-il-crop)")
print("=" * 60)

step_index = json.load(open('splits/step_index.json'))
correct, total = 0, 0
errors_by_class = {1: [], 2: [], 3: [], 4: []}

for L_str, entries in step_index.items():
    L = int(L_str)
    entries_forward = [e for e in entries if e.get('direction') == 'forward']
    for entry in entries_forward[:50]:
        data = np.load(entry['donor_path'])
        n_steps, _ = count_steps_fixed_boundary(data, entry['start_frame'], entry['end_frame'])
        error = n_steps - L
        errors_by_class[L].append(error)
        total += 1
        if error == 0:
            correct += 1

print(f"\nAccuratezza sulla ground truth: {correct}/{total} ({100*correct/total:.1f}%)")
for L, errs in errors_by_class.items():
    if not errs:
        continue
    errs = np.array(errs)
    print(f"Classe {L}: media errore={errs.mean():.2f}, esatto={100*np.mean(errs==0):.1f}%, n={len(errs)}")


print("\n" + "=" * 60)
print("VERIFICA 2: Video generati (fix liftoff implicito al frame 0)")
print("=" * 60)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_npy', default=None)
parser.add_argument('--foot_detect_thres', type=float, default=0.002)
args, _ = parser.parse_known_args()

if args.results_npy:
    from count_with_tolerance import compute_foot_contact_from_joints
    from count_generated_steps import find_candidate_step_windows_from_contact

    raw = np.load(args.results_npy, allow_pickle=True).item()
    motion = raw['motion']
    lengths = raw['lengths']

    for i in range(min(10, motion.shape[0])):
        joints_i = motion[i].transpose(2, 0, 1)[:int(lengths[i])]
        contact_i = compute_foot_contact_from_joints(joints_i, args.foot_detect_thres)

        candidates_old = find_candidate_step_windows_from_contact(contact_i)
        n_steps_old = sum(1 for c in candidates_old if is_step_valid(
            joints=joints_i, liftoff=c['liftoff'], touchdown=c['touchdown'],
            foot=c['foot'], min_horizontal_m=0.15, min_height_m=0.02))

        n_steps_new, candidates_new = count_valid_steps_with_edge_fix(joints_i, contact_i)

        print(f"Video {i}: SENZA fix={n_steps_old}, CON fix={n_steps_new}, "
              f"candidati_con_fix={len(candidates_new)}")
else:
    print("Nessun --results_npy specificato, salto la Verifica 2.")
    print("Rilancia con: python scripts/verify_dataset_fixed.py --results_npy PATH")