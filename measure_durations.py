import json
import re
import os
from collections import defaultdict

# Flag che consideriamo "minori" (non bloccano l'uso del file): al momento solo
# il cambio di orientamento finale ("turns to"/"turn to"), che aggiunge pochi
# frame e non altera sostanzialmente il conteggio dei passi.
MINOR_ONLY_PATTERNS = ['turns to', 'turn to']


def extract_source_id(filename):
    """
    Da 'cut_119_004287.npy' o 'cut_156_M012725.npy' estrae l'ID del clip
    sorgente HumanML3D ('004287' o 'M012725'), rimuovendo il prefisso
    'cut_<N>_' se presente. Se il file non ha prefisso 'cut_', ritorna il nome
    base senza estensione (probabilmente e' gia' un file HumanML3D originale,
    non un segmento tagliato).
    """
    base = filename.replace('.npy', '')
    m = re.match(r'^cut_\d+_(.+)$', base)
    if m:
        return m.group(1)
    return base


def is_minor_only(flags):
    """True se TUTTI i flag di questo file sono del tipo 'minore' (es. solo
    COMPOUND_ACTION(turns to)), quindi il file puo' essere considerato
    utilizzabile nonostante il flag."""
    if not flags:
        return False
    for fl in flags:
        # un flag e' 'minore' solo se e' un COMPOUND_ACTION i cui hit sono
        # ESCLUSIVAMENTE parole della lista minor
        if fl.startswith('COMPOUND_ACTION('):
            inner = fl[len('COMPOUND_ACTION('):-1]
            hit_words = [w.strip() for w in inner.split(',')]
            if not all(w in MINOR_ONLY_PATTERNS for w in hit_words):
                return False
        else:
            # STAIR_CONTAMINATION, COUNT_MISMATCH, NO_COUNT_FOUND -> mai minori
            return False
    return True


def categorize(report_path="audit_report.json", out_dir="."):
    with open(report_path, 'r', encoding='utf-8') as fr:
        report = json.load(fr)

    discard_list = []       # da scartare / sostituire con nuovo materiale
    minor_list = []         # flaggati ma utilizzabili (azione minima)
    source_issue_count = defaultdict(lambda: {'total_cuts': 0, 'flagged_cuts': 0, 'reasons': set()})

    for folder_name, folder_report in report.items():
        for f in folder_report['files']:
            fname = f['file']
            source_id = extract_source_id(fname)
            source_issue_count[source_id]['total_cuts'] += 1

            if f['is_clean']:
                continue

            if is_minor_only(f['flags']):
                minor_list.append({'folder': folder_name, 'file': fname,
                                   'caption': f['caption'], 'flags': f['flags']})
            else:
                discard_list.append({'folder': folder_name, 'file': fname,
                                     'caption': f['caption'], 'flags': f['flags']})
                source_issue_count[source_id]['flagged_cuts'] += 1
                for fl in f['flags']:
                    reason = fl.split('(')[0]
                    source_issue_count[source_id]['reasons'].add(reason)

    sources_to_avoid = {
        sid: info for sid, info in source_issue_count.items()
        if info['flagged_cuts'] > 0 and info['flagged_cuts'] == info['total_cuts']
    }
    # sorgenti parzialmente problematiche: solo alcuni segmenti sono da scartare,
    # altri (stesso sorgente, altro range di frame) sono puliti -> il sorgente
    # di per se' potrebbe essere riutilizzabile con un taglio diverso
    sources_partial = {
        sid: info for sid, info in source_issue_count.items()
        if 0 < info['flagged_cuts'] < info['total_cuts']
    }

    print("=" * 70)
    print(f"DA SCARTARE / SOSTITUIRE: {len(discard_list)} file")
    print("=" * 70)
    for item in discard_list:
        print(f"  [{item['folder']}] {item['file']}  {item['flags']}")
        print(f"      \"{item['caption']}\"")

    print()
    print("=" * 70)
    print(f"MINORI (utilizzabili nonostante il flag): {len(minor_list)} file")
    print("=" * 70)
    for item in minor_list:
        print(f"  [{item['folder']}] {item['file']}  {item['flags']}")
        print(f"      \"{item['caption']}\"")

    print()
    print("=" * 70)
    print(f"SORGENTI HUMANML3D DA EVITARE DEL TUTTO (tutti i cut sono problematici): "
          f"{len(sources_to_avoid)}")
    print("=" * 70)
    for sid, info in sorted(sources_to_avoid.items()):
        print(f"  {sid}: {info['flagged_cuts']}/{info['total_cuts']} cut flaggati, "
              f"motivi={sorted(info['reasons'])}")

    if sources_partial:
        print()
        print("=" * 70)
        print(f"SORGENTI PARZIALMENTE PROBLEMATICHE (alcuni cut ok, altri no -> "
              f"forse riutilizzabili con taglio diverso): {len(sources_partial)}")
        print("=" * 70)
        for sid, info in sorted(sources_partial.items()):
            print(f"  {sid}: {info['flagged_cuts']}/{info['total_cuts']} cut flaggati, "
                  f"motivi={sorted(info['reasons'])}")

    # --- scrivi file di testo pronti per il tuo workflow (una riga = un nome file) ---
    discard_path = os.path.join(out_dir, "files_to_discard.txt")
    minor_path = os.path.join(out_dir, "files_minor_keep.txt")
    sources_avoid_path = os.path.join(out_dir, "sources_to_avoid.txt")

    with open(discard_path, 'w', encoding='utf-8') as fw:
        for item in discard_list:
            fw.write(f"{item['folder']}/{item['file']}\n")

    with open(minor_path, 'w', encoding='utf-8') as fw:
        for item in minor_list:
            fw.write(f"{item['folder']}/{item['file']}\n")

    with open(sources_avoid_path, 'w', encoding='utf-8') as fw:
        canonical_ids = sorted(set(
            sid[1:] if sid.startswith('M') and sid[1:].isdigit() else sid
            for sid in sources_to_avoid.keys()
        ))
        for sid in canonical_ids:
            fw.write(f"{sid}\n")

    print(f"\nSalvati:\n  {discard_path}\n  {minor_path}\n  {sources_avoid_path}")

    return discard_list, minor_list, sources_to_avoid, sources_partial


if __name__ == "__main__":
    import sys
    report_path = sys.argv[1] if len(sys.argv) > 1 else "audit_report.json"
    categorize(report_path)