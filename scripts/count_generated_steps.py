#!/usr/bin/env python
"""
Conta i passi VALIDI in ciascun video generato da sample.generate.py e
confronta il conteggio con il numero di passi ATTESO (la classe L con
cui e' stato generato), calcolando quanti video hanno un conteggio
ESATTAMENTE corretto.

FORMATO DI INPUT (results.npy, come prodotto da sample/generate.py):
    {
        'motion': array [bs, njoints, 3, seqlen]  - posizioni XYZ GLOBALI
                  dei giunti, gia' ricostruite (recover_from_ric + rot2xyz)
        'foot_contact': array [bs, 1, seqlen, 4]  - canali di foot-contact
                  GREZZI (259:263 del vettore hml), salvati PRIMA della
                  conversione in xyz, apposta per permettere questo conteggio
        'lengths': array [bs]                     - lunghezza reale
                  (pre-padding) di ciascun sample
        'text': lista di caption
        'num_samples', 'num_repetitions'
    }

RIUSO della metrica di validazione:
    group_into_blocks() e is_step_valid() vengono importate DIRETTAMENTE
    da build_step_index.py, invariate - sono gia' agnostiche rispetto al
    formato dei dati (operano su joints/timeline gia' calcolati), quindi
    la definizione di "passo valido" resta identica a quella usata per
    costruire l'indice di training.

    find_candidate_step_windows() e block_is_genuine_walking(), invece,
    nello script originale leggono direttamente il vettore hml grezzo
    (rispettivamente per il foot-contact e per la traiettoria del root
    via velocita' locali) - dati che qui NON abbiamo piu' nella stessa
    forma, perche' 'motion' e' gia' in coordinate XYZ globali. Sono
    quindi reimplementate in questo script per lavorare sui giunti/
    foot-contact gia' pronti, usando PERO' le stesse soglie (min_horizontal_m,
    min_height_m, min_straightness) e la stessa identica logica.

USO:
    python count_generated_steps.py \
        --results_npy save/my_finetuning_v11_rebalanced/samples_test/results.npy \
        --expected_classes_per_sample 1,2,3,4,1,2,3,4,1,2

    Il numero di valori in --expected_classes_per_sample deve combaciare
    con num_samples (l'ordine e' quello dei sample nel testset, ripetuto
    identico per ogni ripetizione - num_repetitions).
"""
import argparse
import json
import os
import sys

import numpy as np

# Import DIRETTO delle parti di build_step_index.py che sono gia'
# agnostiche rispetto al formato dei dati.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_step_index import group_into_blocks, is_step_valid, FID_L, FID_R

# Per la modalita' automatica (--step_index), serve importare il
# dataloader vero e proprio - la root del progetto deve essere sul path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Rilevamento candidati - versione per foot_contact PRECALCOLATO
# (stessa logica di find_candidate_step_windows in build_step_index.py,
# ma legge da un array (T, 4) gia' isolato invece che da data[:, CONTACT_SLICE])
# ============================================================

def find_candidate_step_windows_from_contact(foot_contact):
    contacts = (foot_contact > 0.5).astype(int)

    results = []
    pending_liftoff = {'left': None, 'right': None}
    is_edge_case = {'left': False, 'right': False}

    last_left = contacts[0][0] | contacts[0][1]
    last_right = contacts[0][2] | contacts[0][3]

    # FIX: liftoff implicito se il piede è già sollevato al frame 0
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


def get_validated_timeline_from_precomputed(joints, foot_contact, min_horizontal_m, min_height_m):
    candidates = find_candidate_step_windows_from_contact(foot_contact)
    for c in candidates:
        if c.get('is_edge_case', False):
            # Passo "di bordo": il liftoff e' avvenuto prima dell'inizio
            # osservabile, quindi la distanza calcolata sarebbe
            # artificialmente sottostimata (si parte da meta' movimento).
            # Verifica SOLO l'altezza, saltando il controllo di distanza.
            c['valid'] = is_step_valid(
                joints=joints, liftoff=c['liftoff'], touchdown=c['touchdown'],
                foot=c['foot'], min_horizontal_m=0.0,  # nessuna soglia minima
                min_height_m=min_height_m,
            )
        else:
            c['valid'] = is_step_valid(
                joints=joints, liftoff=c['liftoff'], touchdown=c['touchdown'],
                foot=c['foot'], min_horizontal_m=min_horizontal_m, min_height_m=min_height_m,
            )
    return candidates


# ============================================================
# block_is_genuine_walking - versione per joints GIA' GLOBALI
# (l'originale ricostruiva la traiettoria del root integrando le
# velocita' locali dal vettore hml grezzo; qui il root e' gia' in
# coordinate globali dentro 'joints', quindi si legge direttamente)
# ============================================================

ROOT_JOINT_IDX = 0  # indice del giunto radice/pelvis nella convenzione a 22 giunti HumanML3D/SMPL


def block_is_genuine_walking_from_joints(joints, block, min_straightness=0.3):
    """
    Stessa logica/soglia di block_is_genuine_walking() dell'originale,
    ma legge la traiettoria XZ del root DIRETTAMENTE da joints
    (gia' globali), invece di ricostruirla da velocita' locali.
    """
    start = block[0]['liftoff']
    end = block[-1]['touchdown']

    traj = joints[start:end + 1, ROOT_JOINT_IDX, [0, 2]]  # (N, 2) - piano XZ
    net_displacement = np.linalg.norm(traj[-1] - traj[0])
    path_length = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
    straightness = net_displacement / path_length if path_length > 1e-6 else 0.0

    return straightness >= min_straightness


# ============================================================
# Conteggio passi per un singolo video generato
# ============================================================

def count_steps_in_generated_video(joints, foot_contact, length,
                                    min_horizontal_m=0.15, min_height_m=0.02,
                                    min_straightness=0.3):
    """
    joints: array (seqlen, njoints, 3) - posizioni globali gia' ricostruite
    foot_contact: array (seqlen, 4)
    length: lunghezza reale (pre-padding) di questo sample - usata per
            tagliare via eventuale coda di padding prima del conteggio
    """
    joints = joints[:length]
    foot_contact = foot_contact[:length]

    timeline = get_validated_timeline_from_precomputed(
        joints, foot_contact, min_horizontal_m, min_height_m
    )
    blocks = group_into_blocks(timeline)  # RIUSATA identica dall'originale

    genuine_blocks = [
        b for b in blocks
        if block_is_genuine_walking_from_joints(joints, b, min_straightness)
    ]

    n_steps_total = sum(len(b) for b in genuine_blocks)
    detail = [len(b) for b in genuine_blocks]

    return n_steps_total, len(genuine_blocks), detail


# ============================================================
# Derivazione AUTOMATICA delle classi attese - stesso codice,
# stesso seed (123), di generate.py / FixedEvalDatasetLoader.
# Garantisce un ordine IDENTICO a quello usato in fase di generazione,
# senza bisogno di scrivere a mano centinaia di valori.
# ============================================================

def derive_expected_classes_auto(step_index_path, split):
    """
    Istanzia lo stesso dataset (stessa classe, stesso seed) che
    generate.py usa per costruire 'texts' quando --split e' specificato,
    e ne estrae il campo 'L' di ogni item, NELL'ORDINE ESATTO in cui
    generate.py li ha poi passati al modello (test_dataset[i] per
    i in range(len(test_dataset))).
    """
    from data_loaders.get_data import get_dataset_loader

    loader = get_dataset_loader(
        name='custom_dataset', batch_size=1, num_frames=None,
        split=split, device='cpu', step_index_path=step_index_path,
    )
    dataset = loader.dataset

    if not hasattr(dataset, 'fixed_items'):
        raise SystemExit(
            "Il dataset ottenuto non ha l'attributo 'fixed_items' - "
            "probabile split='train' (CustomDatasetLoader), che non e' "
            "quello usato per la valutazione. Passa --split test o --split val."
        )

    return [int(dataset.fixed_items[i]['L']) for i in range(len(dataset))]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_npy', required=True,
                         help="Path al file results.npy prodotto da sample.generate")
    parser.add_argument('--expected_classes_per_sample', default=None,
                         help="[Modalita' MANUALE] Classi attese, UNA per ogni sample nel "
                              "testset, nell'ordine, separate da virgola (es. '1,2,3,4'). "
                              "Viene ripetuta automaticamente per ogni ripetizione. "
                              "Alternativa a --step_index.")
    parser.add_argument('--step_index', default=None,
                         help="[Modalita' AUTOMATICA, CONSIGLIATA] Path a step_index.json - "
                              "ricostruisce l'ordine esatto delle classi attese istanziando "
                              "lo stesso dataset/seed usato da generate.py. "
                              "Va lanciato dalla ROOT del progetto (non da scripts/), "
                              "altrimenti gli import di data_loaders falliscono.")
    parser.add_argument('--split', default='test',
                         help="Split da usare in modalita' automatica (default 'test')")
    parser.add_argument('--min_horizontal_m', type=float, default=0.15,
                         help="Deve combaciare con quello usato in build_step_index.py")
    parser.add_argument('--min_height_m', type=float, default=0.02,
                         help="Deve combaciare con quello usato in build_step_index.py")
    parser.add_argument('--min_straightness', type=float, default=0.3,
                         help="Deve combaciare con quello usato in build_step_index.py")
    args = parser.parse_args()

    raw = np.load(args.results_npy, allow_pickle=True).item()

    motion = raw['motion']          # [bs, njoints, 3, seqlen]
    foot_contact = raw['foot_contact']  # [bs, 1, seqlen, 4]
    lengths = raw['lengths']        # [bs]
    num_samples = raw['num_samples']
    num_repetitions = raw['num_repetitions']

    if args.step_index is not None:
        print(f"Ricostruzione automatica classi attese da '{args.step_index}' "
              f"(split='{args.split}')...")
        expected_per_sample = derive_expected_classes_auto(args.step_index, args.split)
    elif args.expected_classes_per_sample is not None:
        expected_per_sample = [int(x) for x in args.expected_classes_per_sample.split(',')]
    else:
        raise SystemExit("Serve --step_index (consigliato) oppure --expected_classes_per_sample")

    if len(expected_per_sample) != num_samples:
        raise SystemExit(
            f"Le classi attese ricostruite sono {len(expected_per_sample)}, "
            f"ma num_samples nel results.npy e' {num_samples}. "
            f"Verifica di aver usato lo stesso --step_index e --split con cui "
            f"e' stato generato results.npy."
        )
    expected_classes = expected_per_sample * num_repetitions  # ripetuta per ogni rep

    bs = motion.shape[0]
    print(f"Totale video nel file: {bs} (num_samples={num_samples} x "
          f"num_repetitions={num_repetitions})\n")

    results = []
    for i in range(bs):
        # motion[i]: (njoints, 3, seqlen) -> serve (seqlen, njoints, 3)
        joints_i = motion[i].transpose(2, 0, 1)
        # foot_contact[i]: (1, seqlen, 4) -> (seqlen, 4)
        contact_i = foot_contact[i].squeeze(0)
        length_i = int(lengths[i])
        expected_L = expected_classes[i]

        n_steps, n_blocks, detail = count_steps_in_generated_video(
            joints_i, contact_i, length_i,
            args.min_horizontal_m, args.min_height_m, args.min_straightness
        )

        match = (n_steps == expected_L)

        results.append({
            'idx': i,
            'rep': i // num_samples,
            'sample_in_rep': i % num_samples,
            'expected_L': expected_L,
            'n_steps_detected': n_steps,
            'n_genuine_blocks': n_blocks,
            'blocks_detail': detail,
            'match': match,
        })

    # ============================================================
    # Report per-video
    # ============================================================
    print(f"{'Idx':<5} {'Rep':<4} {'Sample':<7} {'Atteso':<7} {'Rilevato':<9} "
          f"{'Blocchi':<8} {'Match':<8} Dettaglio blocchi")
    print("-" * 90)
    for r in results:
        match_str = "OK" if r['match'] else "MISMATCH"
        print(f"{r['idx']:<5} {r['rep']:<4} {r['sample_in_rep']:<7} {r['expected_L']:<7} "
              f"{r['n_steps_detected']:<9} {r['n_genuine_blocks']:<8} {match_str:<8} "
              f"{r['blocks_detail']}")

    # ============================================================
    # Riepilogo aggregato
    # ============================================================
    n_match = sum(1 for r in results if r['match'])
    n_multi_block = sum(1 for r in results if r['n_genuine_blocks'] > 1)
    n_zero_block = sum(1 for r in results if r['n_genuine_blocks'] == 0)

    print("\n=== Riepilogo ===")
    accuracy = 100.0 * n_match / len(results)
    print(f"Conteggio corretto (passi rilevati == classe attesa): "
          f"{n_match}/{len(results)} ({accuracy:.1f}%)")
    print(f"Video con piu' di un blocco genuino (movimento 'spezzato'): {n_multi_block}")
    print(f"Video con zero blocchi genuini (nessun passo valido rilevato): {n_zero_block}")

    print("\n--- Breakdown per classe attesa ---")
    classes = sorted(set(r['expected_L'] for r in results))
    for L in classes:
        subset = [r for r in results if r['expected_L'] == L]
        match_subset = sum(1 for r in subset if r['match'])
        avg_detected = sum(r['n_steps_detected'] for r in subset) / len(subset)
        acc = 100.0 * match_subset / len(subset)
        print(f"  Classe {L}: {match_subset}/{len(subset)} corretti ({acc:.1f}%), "
              f"media passi rilevati = {avg_detected:.2f}")


if __name__ == '__main__':
    main()