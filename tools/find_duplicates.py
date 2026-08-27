"""
Trova i clip byte-identici in un dataset a classi di passi e li raggruppa per
causa (copia _gen, finestra _aug ripetuta, ritagli cut_ sovrapposti).

    python tools/find_duplicates.py dataset_ris
    python tools/find_duplicates.py dataset_ris --kind cut --limit 20
    python tools/find_duplicates.py dataset_ris --keep-list dedup_train.txt

--keep-list scrive un elenco di .npy con UN SOLO rappresentante per gruppo
duplicato, pronto da usare come train.txt deduplicato.
"""
import argparse
import glob
import hashlib
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_index import origin_kind, read_splits  # noqa: E402

CAUSES = {
    frozenset({'orig', 'gen'}): 'gen == originale (copia esatta)',
    frozenset({'gen'}): 'gen == gen (copie esatte fra loro)',
    frozenset({'orig', 'aug'}): 'aug == originale (copia esatta)',
    frozenset({'aug'}): 'aug == aug (stessa finestra temporale)',
    frozenset({'cut'}): 'cut == cut (ritagli sovrapposti dello stesso clip)',
    frozenset({'orig'}): 'due clip originali identici',
}


def cause_of(paths):
    kinds = frozenset(origin_kind(p) for p in paths)
    return CAUSES.get(kinds, ' + '.join(sorted(kinds)))


def caption(npy):
    txt = npy[:-4] + '.txt'
    if os.path.exists(txt):
        with open(txt, encoding='utf-8', errors='replace') as f:
            return f.read().strip()
    return ''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', nargs='?', default='dataset_ris')
    ap.add_argument('--kind', choices=['gen', 'aug', 'cut', 'orig'],
                    help='mostra solo i gruppi che coinvolgono questo tipo di file')
    ap.add_argument('--limit', type=int, default=15, help='gruppi da stampare per causa')
    ap.add_argument('--keep-list', help='scrivi un elenco .npy deduplicato in questo file')
    args = ap.parse_args()

    root = os.path.normpath(args.root)
    splits = read_splits(root)

    by_hash = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, '*_step', '*.npy'))):
        by_hash[hashlib.md5(np.load(f).tobytes()).hexdigest()].append(f)

    groups = [(h, g) for h, g in by_hash.items() if len(g) > 1]
    total = sum(len(g) for g in by_hash.values())
    redundant = sum(len(g) - 1 for _, g in groups)

    by_cause = defaultdict(list)
    for h, g in groups:
        by_cause[cause_of(g)].append((h, g))

    print(f"\n{root}: {total} clip, {len(by_hash)} contenuti distinti, "
          f"{redundant} file ridondanti ({redundant / total * 100:.1f}%) "
          f"in {len(groups)} gruppi\n")

    summary = sorted(by_cause.items(), key=lambda kv: -sum(len(g) - 1 for _, g in kv[1]))
    for cause, gs in summary:
        print(f"  {sum(len(g) - 1 for _, g in gs):>4} file ridondanti "
              f"in {len(gs):>4} gruppi   {cause}")

    for cause, gs in summary:
        shown = [(h, g) for h, g in gs
                 if not args.kind or any(origin_kind(p) == args.kind for p in g)]
        if not shown:
            continue
        print(f"\n{'=' * 78}\n{cause}   ({len(shown)} gruppi)\n{'=' * 78}")
        for h, g in sorted(shown, key=lambda x: -len(x[1]))[:args.limit]:
            frames = np.load(g[0], mmap_mode='r').shape[0]
            print(f"\n  md5 {h[:16]}  {len(g)} file identici  ({frames} frame)")
            for p in g:
                sp = splits.get(os.path.normpath(p), '?')
                print(f"    [{sp:<5}] {os.path.relpath(p, root)}")
                print(f"             \"{caption(p)}\"")
        if len(shown) > args.limit:
            print(f"\n  ... e altri {len(shown) - args.limit} gruppi "
                  f"(alza --limit per vederli)")

    if args.keep_list:
        keep = []
        for h, g in by_hash.items():
            # tieni il rappresentante piu' "originale": orig > cut > gen/aug
            rank = {'orig': 0, 'cut': 1, 'aug': 2, 'gen': 3}
            keep.append(sorted(g, key=lambda p: (rank[origin_kind(p)], p))[0])
        keep.sort()
        with open(args.keep_list, 'w', encoding='utf-8') as f:
            for p in keep:
                f.write(f'{p}\n')
        print(f"\nElenco deduplicato ({len(keep)} clip) scritto in {args.keep_list}")
        print('  distribuzione per classe:',
              dict(Counter(os.path.basename(os.path.dirname(p)) for p in keep)))


if __name__ == '__main__':
    main()
