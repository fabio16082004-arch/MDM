#!/usr/bin/env python
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

# Path del progetto MDM corrente
SCRIPT_DIR = Path(__file__).resolve().parent
MDM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MDM_ROOT))

from count_generated_steps import count_steps_in_generated_video

# fid_l/fid_r: indici (caviglia, punta) per piede sinistro/destro nello
# scheletro a 22 giunti HumanML3D - stessi valori confermati in
# data_loaders/humanml/scripts/motion_process.py (riga 462) e coerenti
# con FID_L, FID_R = [7, 10], [8, 11] già usati in tools/dataset_index.py
FID_L = [7, 10]
FID_R = [8, 11]

# Path del progetto TMR (cerca ~/TMRPlusPlus o percorso relativo parallelo)
USER_HOME = Path.home()
TMR_ROOT = USER_HOME / "TMRPlusPlus"

if str(TMR_ROOT) not in sys.path:
    sys.path.insert(0, str(TMR_ROOT))

try:
    from src.config import read_config
    from src.load import load_model_from_cfg
    from src.data.collate import collate_x_dict
    from hydra.utils import instantiate
    import src.prepare  # noqa - registra i resolver Hydra necessari
except ImportError as e:
    raise ImportError(
        f"Impossibile importare TMR da {TMR_ROOT}. "
        "Verifica che la cartella ~/TMRPlusPlus esista."
    ) from e


def calculate_fid(act1, act2):
    """Calcola la Frechet Inception Distance tra due matrici di feature."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def calculate_tmr_matching_score(text_emb, motion_emb):
    """Calcola la similarità coseno media per ogni coppia (testo, movimento generato)."""
    text_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    motion_norm = motion_emb / (np.linalg.norm(motion_emb, axis=1, keepdims=True) + 1e-8)
    cosine_sim = (np.sum(text_norm * motion_norm, axis=1) + 1) / 2
    return float(np.mean(cosine_sim))


def move_x_dict_to_device(x_dict, device):
    """
    collate_x_dict restituisce un dizionario Python (non un oggetto
    PyTorch), quindi non ha il metodo .to() nativo dei tensori - va
    spostato sul device manualmente, spostando solo i valori che sono
    effettivamente tensori.
    """
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in x_dict.items()
    }


def foot_detect(positions, thres):
    """
    Replica ESATTA (stessa formula, stessi indici) della funzione
    foot_detect() annidata in data_loaders/humanml/scripts/
    motion_process.py (righe 47-58) - non importabile direttamente
    perche' definita come funzione nested, con fid_l/fid_r presi per
    closure dai parametri della funzione esterna extract_features().

    Rileva il contatto col suolo puramente dalla VELOCITA' (spostamento
    al quadrato sotto soglia tra frame consecutivi) - non dall'altezza
    assoluta, coerente con l'implementazione ufficiale del preprocessing
    HumanML3D (il ramo commentato con heightfactor non e' mai stato
    attivato nel codice originale).
    """
    velfactor = np.array([thres, thres])

    feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
    feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
    feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
    feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
    feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)

    return feet_l, feet_r


def compute_foot_contact_from_joints(joints, thres):
    """
    Deriva il segnale di foot-contact GEOMETRICAMENTE dalle posizioni
    dei giunti generati, usando foot_detect() - invece di fidarsi del
    canale di foot-contact PREDETTO dal modello (uno dei 263 output
    della rete, senza garanzia di coerenza interna con le posizioni
    effettive dei giunti che il modello ha generato per quello stesso
    frame).

    joints: (T, njoints, 3)
    Ritorna: (T-1, 4) nello stesso ordine di CONTACT_SLICE
             (caviglia sx, punta sx, caviglia dx, punta dx).
    """
    feet_l, feet_r = foot_detect(joints, thres)
    return np.concatenate([feet_l, feet_r], axis=1)


def detect_cutoff(joints, eps=0.01, min_plateau_frames=15, search_start=0):
    """
    Rileva il cutoff (il frame in cui il vero movimento finisce e
    inizia la chiusura/quiete sintetica) cercando la PRIMA finestra di
    min_plateau_frames consecutivi in cui la velocita' media del root
    (posizione XZ) resta sotto la soglia eps.

    Parametri di default CALIBRATI (Punto 4 della metodologia) tramite
    grid search su un dataset di calibrazione con t_cut noto (build_
    calibration_dataset.py + calibrate_cutoff_detector.py): MAE=2.66
    frame, bilanciato tra tutte le classi (2.44-3.01 frame). Sostituiscono
    la precedente euristica scelta a mano (quiet_check_frames=15,
    max_quiet_speed=0.02), mai validata e che empiricamente peggiorava
    le metriche invece di migliorarle.

    Ritorna len(joints) se nessun plateau viene rilevato (nessun
    arresto, l'intera sequenza viene considerata movimento valido).
    """
    T = joints.shape[0]
    root_xz = joints[:, 0, [0, 2]]  # giunto 0 = root/pelvis, piano XZ

    for start in range(search_start, T - min_plateau_frames):
        end = start + min_plateau_frames
        window = root_xz[start:end]
        speeds = np.linalg.norm(np.diff(window, axis=0), axis=1)
        avg_speed = speeds.mean() if len(speeds) > 0 else 0.0
        if avg_speed < eps:
            return start

    return T


def count_steps_with_calibrated_cutoff(joints, foot_contact, length,
                                        min_horizontal_m=0.15, min_height_m=0.02,
                                        min_straightness=0.3,
                                        cutoff_eps=0.01, cutoff_min_plateau=15,
                                        min_frames_for_analysis=3):
    """
    Come count_steps_in_generated_video(), ma limita il conteggio ai
    frame PRIMA del cutoff stimato con parametri CALIBRATI (Punto 7
    della metodologia) - i frame nella fase di chiusura/quiete SLERP
    non vengono considerati nel conteggio dei passi.

    FIX caso limite: se il cutoff stimato e' cosi' precoce che la
    sequenza troncata risulta troppo corta per essere analizzata
    (es. il modello genera un movimento gia' quasi fermo fin dai
    primissimi frame, tipico di checkpoint iniziali/poco addestrati),
    ritorna 0 passi invece di propagare un IndexError da un array di
    foot-contact vuoto.
    """
    cutoff = detect_cutoff(joints, cutoff_eps, cutoff_min_plateau)

    if cutoff < min_frames_for_analysis:
        return 0

    joints_truncated = joints[:cutoff]
    contact_truncated = foot_contact[:max(0, cutoff - 1)]
    length_truncated = min(length, cutoff)

    if contact_truncated.shape[0] == 0:
        return 0

    n_steps, _, _ = count_steps_in_generated_video(
        joints_truncated, contact_truncated, length_truncated,
        min_horizontal_m, min_height_m, min_straightness
    )
    return n_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_npy', required=True)
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--min_horizontal_m', type=float, default=0.15)
    parser.add_argument('--min_height_m', type=float, default=0.02)
    parser.add_argument('--min_straightness', type=float, default=0.3)
    parser.add_argument('--foot_detect_thres', type=float, default=0.002,
                        help='Soglia (spostamento al quadrato, in m^2) per '
                             'foot_detect - valore confermato in '
                             'motion_process.py (riga 60, alternativa '
                             'esplicita a feet_thre commentata nel codice '
                             'originale del preprocessing HumanML3D).')
    parser.add_argument('--cutoff_eps', type=float, default=0.01,
                        help='Soglia di velocità per il rilevamento del cutoff '
                             '(calibrata tramite grid search, Punto 4 della '
                             'metodologia: MAE=2.66 frame).')
    parser.add_argument('--cutoff_min_plateau', type=int, default=15,
                        help='Frame minimi di plateau per il rilevamento del '
                             'cutoff (calibrato: 15).')

    # Percorso di default impostato dentro ~/TMRPlusPlus/models/...
    default_tmr_path = str(TMR_ROOT / "models" / "tmr_humanml3d_guoh3dfeats")
    parser.add_argument('--tmr_path', default=default_tmr_path,
                        help='Percorso assoluto o relativo dei pesi di TMR')
    parser.add_argument('--ckpt_name', default='last',
                        help='Nome del checkpoint da caricare (default: last)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # FIX path relativi: la configurazione interna di TMR (es.
    # data.text_to_token_emb.path) usa path RELATIVI alla cartella
    # ~/TMRPlusPlus (dove il repository si aspetta di essere eseguito).
    # Se lanciamo questo script da un'altra cartella (motion-diffusion-
    # model-cpy), quei path relativi puntano nel posto sbagliato. La
    # soluzione: rendere ASSOLUTI tutti i path passati da riga di
    # comando ADESSO (prima di qualsiasi os.chdir), cosi' restano
    # validi indipendentemente da quale sia la cartella di lavoro attiva
    # nei passi successivi.
    args.results_npy = os.path.abspath(args.results_npy)
    args.step_index = os.path.abspath(args.step_index)
    args.tmr_path = os.path.abspath(args.tmr_path)

    # 1. Caricamento Dataset
    from data_loaders.get_data import get_dataset_loader
    loader = get_dataset_loader(
        name='custom_dataset', batch_size=1, num_frames=None,
        split=args.split, device='cpu', step_index_path=args.step_index,
    )
    dataset = loader.dataset
    expected_per_sample = [int(dataset.fixed_items[i]['L']) for i in range(len(dataset))]

    # 2. Caricamento Risultati Generati
    raw = np.load(args.results_npy, allow_pickle=True).item()
    motion = raw['motion']  # posizioni XYZ (bs, njoints, 3, seqlen) - usato per il conteggio passi
    foot_contact = raw['foot_contact']
    lengths = raw['lengths']
    num_samples = raw['num_samples']
    num_repetitions = raw['num_repetitions']

    # FIX: TMR si aspetta il formato Guo 263-dim (velocita' root, RIC,
    # rotazioni, foot-contact), NON le posizioni XYZ gia' ricostruite
    # in 'motion' - serve il campo 'guo_features', salvato da
    # sample/generate.py PRIMA della conversione recover_from_ric.
    if 'guo_features' not in raw:
        raise SystemExit(
            "Il file results.npy non contiene 'guo_features' - e' stato "
            "generato prima del fix a sample/generate.py che salva il "
            "vettore Guo 263-dim pre-conversione XYZ. Rigenera i risultati "
            "con lo script aggiornato prima di usare questa valutazione TMR."
        )
    guo_features = raw['guo_features']  # (bs, 1, seqlen, 263)

    if len(expected_per_sample) != num_samples:
        raise SystemExit(f"Mismatch: {len(expected_per_sample)} classi attese vs "
                         f"num_samples={num_samples} nel results.npy")

    expected_classes = expected_per_sample * num_repetitions
    bs = motion.shape[0]

    # 3. Caricamento modello TMR (pattern corretto, come in
    # encode_text.py/encode_motion.py del repository TMR++)
    print(f"\nCaricamento TMR da: {args.tmr_path} su dispositivo: {args.device}...")

    # FIX: la configurazione di TMR usa path relativi (es.
    # "datasets/annotations/humanml3d/...") assunti relativi alla
    # cartella del repository TMRPlusPlus stesso. Ci spostiamo li'
    # temporaneamente prima di istanziare i componenti che leggono
    # quei path, poi torniamo alla cartella originale.
    original_cwd = os.getcwd()
    os.chdir(str(TMR_ROOT))

    tmr_cfg = read_config(args.tmr_path)
    tmr = load_model_from_cfg(tmr_cfg, args.ckpt_name, eval_mode=True, device=args.device)

    # Il text model (tokenizzatore/embedder testuale) e il normalizzatore
    # del movimento sono componenti SEPARATE dal modello TMR stesso -
    # vanno istanziate dalla configurazione, esattamente come fanno gli
    # script ufficiali del repository.
    text_model = instantiate(tmr_cfg.data.text_to_token_emb, device=args.device)
    motion_normalizer = instantiate(tmr_cfg.data.motion_loader.normalizer)

    os.chdir(original_cwd)  # torniamo alla cartella originale per il resto dello script

    if 'text' in raw:
        texts = raw['text']
    else:
        texts = [dataset.fixed_items[i % len(dataset)].get('text', '') for i in range(bs)]

    # 4. Estrazione Feature TMR ed Errori Step
    results_by_class = defaultdict(list)
    gt_motions_list = []

    for i in range(len(dataset)):
        item = dataset[i]
        gt_motion = item[0] if isinstance(item, (list, tuple)) else item['motion']
        gt_motions_list.append(gt_motion)

    print("Estrazione embedding latenti con TMR in corso...")
    with torch.inference_mode():
        # --- Testo ---
        # text_model tokenizza/embedda le caption grezze; collate_x_dict
        # impacchetta il risultato nel formato a dizionario atteso da encode()
        text_x_dict = collate_x_dict(text_model(texts))
        text_x_dict = move_x_dict_to_device(text_x_dict, args.device)
        z_text = tmr.encode(text_x_dict, sample_mean=True).cpu().numpy()

        # --- Movimento generato dal modello ---
        # Usa guo_features (formato Guo 263-dim, pre-conversione XYZ),
        # NON motion (che contiene posizioni XYZ già ricostruite,
        # incompatibili con il normalizzatore/encoder di TMR).
        # guo_features ha forma (bs, 1, seqlen, 263): la dimensione
        # singleton (asse 1) va rimossa prima di passare la sequenza
        # al normalizzatore.
        gen_x_dicts = []
        for i in range(bs):
            true_len = int(lengths[i])
            # FIX: il buffer generato ha sempre la stessa dimensione fissa
            # (es. 120 frame), ma solo i primi 'true_len' sono i frame
            # realmente validi secondo la logica di masking del training.
            # Passare l'intero buffer con una "length" dichiarata più corta
            # crea un mismatch tra dimensione del tensore e maschera attesa
            # da TMR - tagliamo quindi il tensore ESPLICITAMENTE alla sua
            # vera lunghezza, coerentemente con come sono già costruiti i
            # gt_motions_list (sempre già tagliati alla loro lunghezza vera).
            m = torch.from_numpy(guo_features[i]).float().squeeze(0)[:true_len]  # (true_len, 263)
            m = motion_normalizer(m)
            gen_x_dicts.append({"x": m, "length": true_len})
        gen_x_dict = collate_x_dict(gen_x_dicts)
        gen_x_dict = move_x_dict_to_device(gen_x_dict, args.device)
        z_gen = tmr.encode(gen_x_dict, sample_mean=True).cpu().numpy()

        # --- Movimento reale (ground truth), lunghezze naturalmente variabili ---
        # collate_x_dict gestisce nativamente il padding/batching di
        # sequenze di lunghezza diversa tramite il campo "length" - non
        # serve piu' un ciclo manuale ne' np.stack.
        gt_x_dicts = []
        for gt_motion in gt_motions_list:
            # FIX: il dataset (FixedEvalDatasetLoader) restituisce i
            # crop gia' come tensori Torch, non come array numpy grezzi -
            # torch.from_numpy() fallisce su un tensore gia' esistente.
            # Gestiamo entrambi i casi per robustezza.
            if torch.is_tensor(gt_motion):
                m = gt_motion.float()
            else:
                m = torch.from_numpy(gt_motion).float()
            m = motion_normalizer(m)
            gt_x_dicts.append({"x": m, "length": len(m)})
        gt_x_dict = collate_x_dict(gt_x_dicts)
        gt_x_dict = move_x_dict_to_device(gt_x_dict, args.device)
        z_gt = tmr.encode(gt_x_dict, sample_mean=True).cpu().numpy()

    # 5. Conteggio Step
    # FIX: il segnale di foot-contact NON viene piu' letto dal canale
    # predetto dal modello (raw['foot_contact']) - viene invece
    # calcolato GEOMETRICAMENTE dalle posizioni dei giunti generati,
    # con la stessa funzione ufficiale foot_detect() usata nel
    # preprocessing HumanML3D. Questo evita di fidarsi di un canale
    # che il modello potrebbe generare in modo incoerente con le
    # posizioni effettive dei giunti nello stesso frame (nessun
    # vincolo esplicito lega i due segnali nell'output della rete).
    for i in range(bs):
        joints_i = motion[i].transpose(2, 0, 1)
        contact_i = compute_foot_contact_from_joints(joints_i, args.foot_detect_thres)
        length_i = int(lengths[i])
        expected_L = expected_classes[i]

        n_steps = count_steps_with_calibrated_cutoff(
            joints_i, contact_i, length_i,
            args.min_horizontal_m, args.min_height_m, args.min_straightness,
            cutoff_eps=args.cutoff_eps, cutoff_min_plateau=args.cutoff_min_plateau
        )
        results_by_class[expected_L].append(n_steps - expected_L)

    classes = sorted(results_by_class.keys())

    # 6. Calcolo Metriche TMR
    fid_score = calculate_fid(z_gen, z_gt)
    tmr_score = calculate_tmr_matching_score(z_text, z_gen)

    # 7. Stampa Tabelle e Risultati
    print("\n" + "=" * 80)
    print(f"{'Classe':<8} {'N video':<9} {'Esatto':<10} {'±1 passo':<11} "
          f"{'Errore >=2':<12} {'Esatto+/-1 (cumulativo)'}")
    print("-" * 80)

    tot_exact, tot_off1, tot_off_more, tot_n = 0, 0, 0, 0

    for L in classes:
        errors = results_by_class[L]
        n = len(errors)

        n_exact = sum(1 for e in errors if e == 0)
        n_off1 = sum(1 for e in errors if abs(e) == 1)
        n_off_more = sum(1 for e in errors if abs(e) >= 2)

        pct_exact = 100.0 * n_exact / n
        pct_off1 = 100.0 * n_off1 / n
        pct_off_more = 100.0 * n_off_more / n
        pct_cumulative = 100.0 * (n_exact + n_off1) / n

        print(f"{L:<8} {n:<9} {pct_exact:>6.1f}%   {pct_off1:>7.1f}%   "
              f"{pct_off_more:>8.1f}%    {pct_cumulative:>6.1f}%")

        tot_exact += n_exact
        tot_off1 += n_off1
        tot_off_more += n_off_more
        tot_n += n

    print("-" * 80)
    print(f"{'TOTALE':<8} {tot_n:<9} "
          f"{100.0*tot_exact/tot_n:>6.1f}%   "
          f"{100.0*tot_off1/tot_n:>7.1f}%   "
          f"{100.0*tot_off_more/tot_n:>8.1f}%    "
          f"{100.0*(tot_exact+tot_off1)/tot_n:>6.1f}%")

    print("\n" + "=" * 45)
    print("           METRICHE QUALITÀ TMR              ")
    print("=" * 45)
    print(f" TMR Matching Score (Cosine Sim) : {tmr_score:.4f}")
    print(f" TMR-FID (Fréchet Distance)      : {fid_score:.4f}")
    print("=" * 45)


if __name__ == '__main__':
    main()