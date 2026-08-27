"""
Report testuale sul dataset di camminate a N passi.

    python tools/dataset_stats.py dataset_ris dataset_risv2
    python tools/dataset_stats.py dataset_ris --json out.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_index import build_index, class_steps, compute_stats  # noqa: E402

BAR = '=' * 92


def _row(label, b):
    return (f"{label:<9}{b['n']:>6}{b['unique_motions']:>8}{b['unique_sources']:>10}"
            f"{b['unique_captions']:>7}{b['frames_mean']:>11.1f}{b['sec_mean']:>9.2f}"
            f"{b['sec_median']:>12.2f}{b['sec_min']:>9.2f}{b['sec_max']:>9.2f}"
            f"{b['sec_total'] / 60:>9.1f}")


def _row_m(label, b, n_steps=None):
    per_step = f"{b['foot_mean'] / n_steps:>10.2f}" if n_steps else f"{'':>10}"
    skate = f"{b['skate_mean']:.2f} (+{b['skate_mean'] / b['foot_mean'] * 100:.0f}%)"
    return (f"{label:<9}{b['foot_mean']:>10.2f}{b['foot_median']:>12.2f}"
            f"{b['foot_std']:>9.2f}{b['foot_min']:>9.2f}{b['foot_max']:>9.2f}"
            f"{per_step}{b['airborne_mean'] * 100:>10.0f}%"
            f"{b['foot_all_mean']:>15.2f}{skate:>16}{b['path_mean']:>10.2f}")


def _fmt_class_table(stats):
    hdr = (f"{'classe':<9}{'n':>6}{'uniche':>8}{'sorgenti':>10}{'capt.':>7}"
           f"{'frame med':>11}{'sec med':>9}{'sec med.na':>12}{'sec min':>9}{'sec max':>9}{'tot min':>9}")
    print(hdr)
    print('-' * len(hdr))
    for cls, b in stats['per_class'].items():
        if b is not None:
            print(_row(cls, b))
    print('-' * len(hdr))
    print(_row('TOTALE', stats['overall']))

    # metri percorsi dai PIEDI sul piano di terra: prima le sole fasi di volo,
    # poi la variante che ignora i foot-contact e la loro differenza (skating)
    hdr = (f"{'classe':<9}{'volo m':>10}{'m mediana':>12}{'dev.std':>9}{'m min':>9}"
           f"{'m max':>9}{'m/passo':>10}{'in volo':>11}{'tutti i frame':>15}"
           f"{'scivolamento':>16}{'pelvis':>10}")
    print()
    print(hdr)
    print('-' * len(hdr))
    for cls, b in stats['per_class'].items():
        if b is not None:
            print(_row_m(cls, b, n_steps=class_steps(cls)))
    print('-' * len(hdr))
    print(_row_m('TOTALE', stats['overall']))


def report(root):
    samples, meta = build_index(root)
    stats = compute_stats(samples, meta)
    o = stats['overall']

    print()
    print(BAR)
    print(f"  {root}   ({meta['n_samples']} clip, {meta['fps']} fps, feature vec263 HumanML3D)")
    print(BAR)
    print()
    _fmt_class_table(stats)

    print(f"\nProvenienza clip: {o['origins']}")
    print("  orig = clip HumanML3D intero | cut = segmento ritagliato | "
          "gen/aug = copia aumentata")

    print("\nSplit:")
    for sp, b in stats['per_split'].items():
        per_cls = {c: sum(1 for s in samples if s['split'] == sp and s['cls'] == c)
                   for c in meta['classes']}
        print(f"  {sp:<6} n={b['n']:<5} per classe={per_cls}")
    unassigned = sum(1 for s in samples if s['split'] == '?')
    if unassigned:
        print(f"  (!) {unassigned} clip non compaiono in nessun file di split")

    d = stats['duplicates']
    print(f"\nDuplicati esatti (byte-identici): {d['redundant_files']} file ridondanti "
          f"su {o['n']} ({d['redundant_files'] / o['n'] * 100:.1f}%), "
          f"in {d['groups']} gruppi -> motion realmente distinte: {o['unique_motions']}")

    lk = stats['leakage']
    if lk['eval_total']:
        print(f"\nLeakage val/test ({lk['eval_total']} clip di valutazione):")
        print(f"  identici a un clip di train : {lk['exact']} "
              f"({lk['exact'] / lk['eval_total'] * 100:.1f}%)")
        print(f"  stesso clip HumanML3D sorgente: {lk['same_source']} "
              f"({lk['same_source'] / lk['eval_total'] * 100:.1f}%)")

    print("\nCaption piu' frequenti per classe:")
    for cls, b in stats['per_class'].items():
        print(f"  [{cls}] {b['unique_captions']} caption distinte su {b['n']} clip")
        for cap, k in b['top_captions'][:3]:
            print(f"      {k:>3}x  \"{cap}\"")
    print()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='*', default=['dataset_ris', 'dataset_risv2'],
                    help='cartelle dataset da analizzare')
    ap.add_argument('--json', help='salva il report completo in un file JSON')
    args = ap.parse_args()

    all_stats = {}
    for root in args.roots:
        if not os.path.isdir(root):
            print(f'[SKIP] {root}: non e\' una cartella')
            continue
        all_stats[root] = report(root)

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f'Report salvato in {args.json}')


if __name__ == '__main__':
    main()
