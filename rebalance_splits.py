import json
import random
from collections import defaultdict

STEP_INDEX_PATH = "splits/step_index.json"
OUT_DIR = "splits"
SEED = 42
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

random.seed(SEED)

with open(STEP_INDEX_PATH, encoding="utf-8") as f:
    step_index = json.load(f)

classes = sorted(step_index.keys(), key=lambda x: int(x))

# 1. Per ogni donor, conta quante entries forward ha per ciascuna classe
donor_class_forward = defaultdict(lambda: defaultdict(int))
all_donors = set()

for L, entries in step_index.items():
    for e in entries:
        donor = e["donor_path"]
        all_donors.add(donor)
        if e.get("direction") == "forward":
            donor_class_forward[donor][L] += 1

def base_id(donor_path):
    """Nome file senza estensione e senza il prefisso 'M' del mirroring."""
    name = donor_path.split('/')[-1].replace('.npy', '')
    return name.lstrip('M')

group_of = {donor: base_id(donor) for donor in all_donors}
members_of_group = defaultdict(list)
for donor, group in group_of.items():
    members_of_group[group].append(donor)

donor_class_forward_grouped = defaultdict(lambda: defaultdict(int))
for donor, donor_classes in donor_class_forward.items():
    group = group_of[donor]
    for L, count in donor_classes.items():
        donor_class_forward_grouped[group][L] += count

# Da qui in poi, "all_donors" e "donor_class_forward" operano sui GRUPPI
# (base+mirror uniti), non piu' sui singoli file .npy.
donor_class_forward = donor_class_forward_grouped
all_donors = sorted(members_of_group.keys())  # ordine deterministico prima dello shuffle

# 2. Calcola il totale forward per classe e il target 80/10/10 (in ENTRIES,
#    non donatori) per ciascuno split
class_totals = {L: sum(donor_class_forward[d].get(L, 0) for d in all_donors)
                 for L in classes}

targets = {
    split_name: {L: class_totals[L] * frac for L in classes}
    for split_name, frac in RATIOS.items()
}

print("=== Entries forward totali disponibili per classe ===")
for L in classes:
    print(f"  Classe {L}: {class_totals[L]} entries totali -> "
          f"target train={targets['train'][L]:.0f}  "
          f"val={targets['val'][L]:.0f}  test={targets['test'][L]:.0f}")

# 3. Assegnazione greedy: per ogni donatore (in ordine casuale ma
#    riproducibile), assegna il donatore allo split che ha il maggior
#    "deficit relativo" pesato sulle classi a cui quel donatore contribuisce.
#    Il deficit relativo di uno split per una classe e':
#        (target - assegnato_finora) / target      (0 se target=0)
#    Il donatore va allo split che massimizza la somma dei deficit
#    pesati dal proprio contributo in ciascuna classe - cosi' i donatori
#    "specializzati" su una classe rara vengono smistati con priorita'
#    verso lo split che ne ha piu' bisogno.

current = {split_name: defaultdict(int) for split_name in RATIOS}
splits = {split_name: [] for split_name in RATIOS}

donors_shuffled = all_donors[:]
random.shuffle(donors_shuffled)

# Donatori con almeno un'entry forward vengono processati per primi,
# ordinati dal piu' "specializzato" (poche classi, contributo concentrato)
# al piu' generico, cosi' i casi vincolanti si risolvono per primi.
def specialization_score(d):
    contribs = donor_class_forward[d]
    if not contribs:
        return 0
    n_classes = len(contribs)
    total = sum(contribs.values())
    return total / n_classes  # piu' alto = piu' concentrato su poche classi

donors_with_forward = [d for d in donors_shuffled if donor_class_forward[d]]
donors_without_forward = [d for d in donors_shuffled if not donor_class_forward[d]]
donors_with_forward.sort(key=specialization_score, reverse=True)

for d in donors_with_forward:
    contribs = donor_class_forward[d]
    best_split, best_score = None, None
    for split_name in RATIOS:
        score = 0.0
        for L, count in contribs.items():
            target = targets[split_name][L]
            if target <= 0:
                continue
            deficit = (target - current[split_name][L]) / target
            score += deficit * count
        if best_score is None or score > best_score:
            best_split, best_score = split_name, score
    splits[best_split].append(d)
    for L, count in contribs.items():
        current[best_split][L] += count

# 4. Donatori senza dati forward (inutili per train/eval, ma restano nel
#    donatore fisico) vengono distribuiti rispettando le proporzioni
#    GLOBALI sul numero di donatori, per non sbilanciare troppo le
#    dimensioni finali degli split.
n_total = len(all_donors)
target_val_donors = int(RATIOS["val"] * n_total)
target_test_donors = int(RATIOS["test"] * n_total)

for d in donors_without_forward:
    if len(splits["val"]) < target_val_donors:
        splits["val"].append(d)
    elif len(splits["test"]) < target_test_donors:
        splits["test"].append(d)
    else:
        splits["train"].append(d)

# 5. Scrivi i nuovi file di split - espandendo ogni group_id nei suoi
# membri reali (base + eventuale mirror), cosi' i .txt finali contengono
# i veri donor_path, non gli identificatori di gruppo astratti.
for split_name, groups in splits.items():
    out_path = f"{OUT_DIR}/{split_name}.txt"
    real_donors = []
    for group in groups:
        real_donors.extend(members_of_group[group])
    with open(out_path, "w", encoding="utf-8") as f:
        for d in sorted(real_donors):
            f.write(d + "\n")
    print(f"Scritto {out_path} ({len(groups)} gruppi, {len(real_donors)} file .npy totali)")

# 6. Verifica finale: per ogni classe, entries forward ottenute vs target
print("\n=== VERIFICA FINALE (entries forward: ottenuto vs target 80/10/10) ===")
all_ok = True
for L in classes:
    line = f"  Classe {L}: "
    parts = []
    for split_name in ("train", "val", "test"):
        got = current[split_name][L]
        target = targets[split_name][L]
        ok = got > 0 and (target == 0 or got >= 0.5 * target)  # tolleranza 50%
        all_ok = all_ok and (got > 0)
        marker = "" if ok else " ⚠"
        parts.append(f"{split_name}={got}/{target:.0f}{marker}")
    print(line + "  ".join(parts))

print(f"\nTotale donor: train={len(splits['train'])}  val={len(splits['val'])}  "
      f"test={len(splits['test'])}  (tot={n_total})")

if all_ok:
    print("\n Tutte le classi hanno dati 'forward' in train, val e test.")
else:
    print("\n  Alcune classi non hanno raggiunto il target o sono scoperte "
          "in qualche split - guarda i marker ⚠ sopra.")