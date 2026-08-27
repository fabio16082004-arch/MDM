#!/usr/bin/env python
"""
Verifica, per ogni entry del JSON, che il numero di touchdown rilevati
dentro [core_start, core_end] coincida con L dichiarato.

Fa DUE conteggi distinti, per distinguere due cause diverse di mismatch:

1. Conteggio ISOLATO: applica find_candidate_step_windows direttamente
   sul solo slice data[start_frame:end_frame], SENZA contesto esterno.
   Puo' SOTTOSTIMARE se il liftoff del primo passo cade esattamente al
   bordo del taglio (non osservabile senza il frame precedente) - questo
   e' un ARTEFATTO della tecnica di rilevazione, non un errore nei dati
   (lo avevamo gia' diagnosticato su M003921/002365.npy in precedenza).

2. Conteggio con CONTESTO PIENO: applica find_candidate_step_windows
   sull'INTERO donatore (con tutto il contesto), poi filtra solo i
   touchdown che cadono dentro [core_start, core_end]. Questo e' il
   conteggio "vero" - se non coincide con L, specialmente se e' PIU'
   ALTO di L, e' un errore genuino nella costruzione dell'indice (la
   zona nucleo non dovrebbe MAI contenere piu' di L touchdown, per
   costruzione dei confini di sicurezza).

Interpretazione:
  - isolato < L  e  contesto_pieno == L  -> ARTEFATTO di rilevazione,
    non un errore nei dati (il conteggio isolato ha perso il contesto)
  - contesto_pieno == L                  -> OK, tutto corretto
  - contesto_pieno != L (specialmente >L) -> ERRORE GENUINO, da indagare

USO:
    python verify_footcontact_vs_L.py --step_index splits/step_index_train.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_step_index import find_candidate_step_windows


def count_isolated(data, start_frame, end_frame):
    """Conteggio SENZA contesto: rilevazione applicata solo al ritaglio
    isolato data[start_frame:end_frame]. Soggetto a sottostima se il
    liftoff del primo passo cade esattamente al bordo del taglio."""
    cut = data[start_frame:end_frame]
    candidates = find_candidate_step_windows(cut)
    return len(candidates)


def count_with_full_context(data, core_start, core_end):
    """Conteggio CON contesto: rilevazione sull'INTERO donatore, poi
    filtra solo i touchdown che cadono dentro (core_start, core_end].
    Nota: core_start e' un LIFTOFF, non un touchdown della finestra -
    il confine inferiore e' STRETTAMENTE maggiore (non >=), altrimenti
    un touchdown del passo PRECEDENTE (fuori dalla finestra) che per
    coincidenza cade esattamente su core_start (doppio appoggio naturale,
    un piede atterra mentre l'altro si stacca nello stesso frame)
    verrebbe contato per errore. Il touchdown del PRIMO passo della
    finestra stessa e' sempre strettamente dopo il proprio liftoff,
    quindi questo fix non esclude mai touchdown legittimi."""
    candidates = find_candidate_step_windows(data)
    return sum(1 for c in candidates if core_start < c['touchdown'] <= core_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--limit_per_class', type=int, default=0,
                         help="0 = verifica tutte le entry; altrimenti solo le prime N per classe (piu' veloce)")
    args = parser.parse_args()

    with open(args.step_index, encoding='utf-8') as f:
        step_index = json.load(f)

    stats = defaultdict(lambda: {'ok': 0, 'artefatto': 0, 'errore_genuino': 0, 'n': 0})
    genuine_errors = []

    for L, entries in step_index.items():
        to_check = entries if args.limit_per_class == 0 else entries[:args.limit_per_class]

        for i, e in enumerate(to_check):
            if i % 50 == 0 and i > 0:
                print(f"  classe {L}: {i}/{len(to_check)} verificate...", flush=True)

            try:
                data = np.load(e['donor_path'])
            except Exception as ex:
                print(f"[ERRORE CARICAMENTO] {e['donor_path']}: {ex}")
                continue

            L_int = int(L)
            core_start, core_end = e['core_start'], e['core_end']
            start_frame, end_frame = e['start_frame'], e['end_frame']

            isolated = count_isolated(data, start_frame, end_frame)
            full_context = count_with_full_context(data, core_start, core_end)

            stats[L]['n'] += 1

            if full_context == L_int:
                stats[L]['ok'] += 1
                if isolated < L_int:
                    stats[L]['artefatto'] += 1
            else:
                stats[L]['errore_genuino'] += 1
                genuine_errors.append({
                    'L': L_int, 'donor_path': e['donor_path'],
                    'core_start': core_start, 'core_end': core_end,
                    'isolato': isolated, 'contesto_pieno': full_context,
                })

    print(f"\n=== Riepilogo ===")
    print(f"{'classe':>8} {'n_verificate':>13} {'OK':>6} {'di cui artefatto rilevazione':>30} {'ERRORE GENUINO':>16}")
    for L in sorted(stats.keys(), key=int):
        s = stats[L]
        print(f"{L:>8} {s['n']:>13} {s['ok']:>6} {s['artefatto']:>30} {s['errore_genuino']:>16}")

    if genuine_errors:
        print(f"\n=== {len(genuine_errors)} ERRORI GENUINI (contesto_pieno != L) ===")
        for err in genuine_errors[:20]:
            flag = " <-- PIU' touchdown del previsto, sospetto serio" if err['contesto_pieno'] > err['L'] else ""
            print(f"  L={err['L']} donor={err['donor_path']} core=({err['core_start']},{err['core_end']}) "
                  f"isolato={err['isolato']} contesto_pieno={err['contesto_pieno']}{flag}")
        if len(genuine_errors) > 20:
            print(f"  ... e altri {len(genuine_errors) - 20}")
    else:
        print("\nNessun errore genuino trovato - tutti i mismatch (se presenti) erano solo artefatti di rilevazione isolata.")


if __name__ == '__main__':
    main()