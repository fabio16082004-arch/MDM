#!/usr/bin/env python
"""
Test di TRASFERIBILITA' / CATASTROPHIC FORGETTING: genera la stessa
batteria di prompt (azioni NON correlate al conteggio passi) sia dal
checkpoint fine-tuned sia dal checkpoint ORIGINALE pre-addestrato
HumanML3D, per un confronto visivo diretto - se il fine-tuning ha
degradato la capacita' generale del modello, dovrebbe essere visibile
confrontando le coppie di video fianco a fianco.

I prompt includono sia azioni completamente diverse dal camminare
(salire, sedersi, ballare, ecc.) sia camminate SENZA menzione del
conteggio (per vedere se il modello continua a generare camminate
plausibili anche senza il contesto "N passi" su cui e' stato
specializzato).

USO (dalla ROOT del progetto):
    python test_transfer.py \
        --finetuned_ckpt save/my_finetuning_v18_clean_split/model000025000.pt \
        --original_ckpt save/humanml_trans_enc_512/model000475000.pt \
        --outdir transfer_test
"""
import argparse
import os
import subprocess
import sys

# Prompt di azioni NON legate al conteggio passi - per testare se il
# modello ha "dimenticato" come generare altro dopo il fine-tuning.
PROMPTS = [
    "a person jumps up and down",
    "a person sits down on a chair",
    "a person waves with the right hand",
    "a person dances happily",
    "a person kicks something with the left leg",
    "a person walks forward slowly",           # camminata SENZA conteggio esplicito
    "a person runs quickly",
    "a person turns around and looks back",
    "a person picks up an object from the floor",
    "a person raises both arms above the head",
]


def run_generation(model_path, prompt, outdir, motion_length=4.0, num_samples=1):
    """Lancia sample.generate per un singolo prompt."""
    cmd = [
        "python", "-m", "sample.generate",
        "--model_path", model_path,
        "--dataset", "custom_dataset",
        "--step_index", "splits/step_index.json",
        "--text_prompt", prompt,
        "--motion_length", str(motion_length),
        "--num_samples", str(num_samples),
        "--num_repetitions", "1",
        "--output_dir", outdir,
    ]
    print(f"  Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERRORE] {result.stderr[-1000:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetuned_ckpt', required=True,
                         help="Path al checkpoint fine-tuned (es. il tuo modello finale)")
    parser.add_argument('--original_ckpt', required=True,
                         help="Path al checkpoint ORIGINALE pre-addestrato HumanML3D "
                              "(es. save/humanml_trans_enc_512/model000475000.pt)")
    parser.add_argument('--outdir', default='transfer_test')
    parser.add_argument('--motion_length', type=float, default=4.0)
    parser.add_argument('--prompts', default=None,
                         help="Path opzionale a un file .txt con un prompt per riga, "
                              "per sostituire la lista di default")
    args = parser.parse_args()

    prompts = PROMPTS
    if args.prompts:
        with open(args.prompts, encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

    finetuned_dir = os.path.join(args.outdir, 'finetuned')
    original_dir = os.path.join(args.outdir, 'original')
    os.makedirs(finetuned_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    print(f"=== Test di trasferibilita': {len(prompts)} prompt x 2 checkpoint ===\n")

    for i, prompt in enumerate(prompts):
        safe_name = prompt.replace(' ', '_').replace(',', '')[:40]

        print(f"[{i+1}/{len(prompts)}] Prompt: \"{prompt}\"")

        print(f"  -> Checkpoint FINE-TUNED")
        ft_out = os.path.join(finetuned_dir, f"{i:02d}_{safe_name}")
        run_generation(args.finetuned_ckpt, prompt, ft_out, args.motion_length)

        print(f"  -> Checkpoint ORIGINALE")
        orig_out = os.path.join(original_dir, f"{i:02d}_{safe_name}")
        run_generation(args.original_ckpt, prompt, orig_out, args.motion_length)

        print()

    print(f"=== Fatto ===")
    print(f"Video fine-tuned in: {finetuned_dir}/")
    print(f"Video originali in:  {original_dir}/")
    print(f"\nConfronta le coppie con lo stesso indice/nome tra le due cartelle "
          f"per valutare a occhio se il fine-tuning ha degradato la qualita' "
          f"di generazione su azioni non legate al conteggio dei passi.")


if __name__ == '__main__':
    main()