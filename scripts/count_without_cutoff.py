#!/usr/bin/env python
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

SCRIPT_DIR = Path(__file__).resolve().parent
MDM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MDM_ROOT))

from count_generated_steps import count_steps_in_generated_video

FID_L = [7, 10]
FID_R = [8, 11]

USER_HOME = Path.home()
TMR_ROOT = USER_HOME / "TMRPlusPlus"

if str(TMR_ROOT) not in sys.path:
    sys.path.insert(0, str(TMR_ROOT))

try:
    from src.config import read_config
    from src.load import load_model_from_cfg
    from src.data.collate import collate_x_dict
    from hydra.utils import instantiate
    import src.prepare  # noqa
except ImportError as e:
    raise ImportError(
        f"Could not import TMR from {TMR_ROOT}. "
        "Make sure the ~/TMRPlusPlus directory exists."
    ) from e

GUO_ROOT = MDM_ROOT / "guo_evaluator_bundle"

if str(GUO_ROOT) not in sys.path:
    sys.path.append(str(GUO_ROOT))

try:
    from guo import GuoEvaluator, evaluate_guo
except ImportError as e:
    raise ImportError(
        f"Could not import the Guo evaluator from {GUO_ROOT}."
    ) from e

GUO_TOKEN_CACHE = MDM_ROOT / "assets" / "guo_caption_tokens.json"


def load_guo_tokens(texts):
    if not GUO_TOKEN_CACHE.exists():
        raise SystemExit(
            f"Guo token cache not found ({GUO_TOKEN_CACHE}). Generate it with:\n"
            "  conda activate TMR && python scripts/build_guo_token_cache.py"
        )
    with open(GUO_TOKEN_CACHE, encoding='utf-8') as f:
        cache = json.load(f)

    missing = sorted({t for t in texts if t not in cache})
    if missing:
        raise SystemExit(
            f"{len(missing)} captions are missing from the Guo token cache "
            f"(e.g. {missing[0]!r}). Regenerate the cache with:\n"
            "  conda activate TMR && python scripts/build_guo_token_cache.py "
            "--results_npy <results.npy>"
        )
    return [cache[t] for t in texts]


def load_guo_evaluator(device):
    """Il bundle carica finest.tar con weights_only=True, che l'unpickler
    ristretto di torch 1.13 (env `mdm`) rifiuta."""
    original_load = torch.load

    def patched_load(*a, **kw):
        kw['weights_only'] = False
        return original_load(*a, **kw)

    torch.load = patched_load
    try:
        return GuoEvaluator(device=device)
    finally:
        torch.load = original_load


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def calculate_tmr_matching_score(text_emb, motion_emb):
    text_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    motion_norm = motion_emb / (np.linalg.norm(motion_emb, axis=1, keepdims=True) + 1e-8)
    cosine_sim = (np.sum(text_norm * motion_norm, axis=1) + 1) / 2
    return float(np.mean(cosine_sim))


def move_x_dict_to_device(x_dict, device):
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in x_dict.items()
    }


def foot_detect(positions, thres):
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
    feet_l, feet_r = foot_detect(joints, thres)
    return np.concatenate([feet_l, feet_r], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_npy', required=True)
    parser.add_argument('--step_index', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--min_horizontal_m', type=float, default=0.15)
    parser.add_argument('--min_height_m', type=float, default=0.02)
    parser.add_argument('--min_straightness', type=float, default=0.3)
    parser.add_argument('--foot_detect_thres', type=float, default=0.002)

    default_tmr_path = str(TMR_ROOT / "models" / "tmr_humanml3d_guoh3dfeats")
    parser.add_argument('--tmr_path', default=default_tmr_path)
    parser.add_argument('--ckpt_name', default='last')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--guo_matching_pool', default='32',
                        help="Dimensione dei gruppi per il matching score Guo "
                             "(32 riproduce il protocollo originale; 'none' "
                             "media tutte le coppie senza scartare il resto)")
    parser.add_argument('--guo_batch_size', type=int, default=32)
    parser.add_argument('--guo_seed', type=int, default=42)
    args = parser.parse_args()

    guo_pool = (None if args.guo_matching_pool.lower() in ('none', 'null')
                else int(args.guo_matching_pool))

    args.results_npy = os.path.abspath(args.results_npy)
    args.step_index = os.path.abspath(args.step_index)
    args.tmr_path = os.path.abspath(args.tmr_path)

    from data_loaders.get_data import get_dataset_loader
    loader = get_dataset_loader(
        name='custom_dataset', batch_size=1, num_frames=None,
        split=args.split, device='cpu', step_index_path=args.step_index,
        shuffle=False,
    )
    dataset = loader.dataset
    expected_per_sample = [int(dataset.fixed_items[i]['L']) for i in range(len(dataset))]

    raw = np.load(args.results_npy, allow_pickle=True).item()
    motion = raw['motion']
    foot_contact = raw['foot_contact']
    lengths = raw['lengths']
    num_samples = raw['num_samples']
    num_repetitions = raw['num_repetitions']

    if 'guo_features' not in raw:
        raise SystemExit("The results.npy file does not contain 'guo_features'.")
    guo_features = raw['guo_features']

    if len(expected_per_sample) != num_samples:
        raise SystemExit(f"Mismatch: {len(expected_per_sample)} expected classes vs "
                         f"num_samples={num_samples} in results.npy")

    expected_classes = expected_per_sample * num_repetitions
    bs = motion.shape[0]

    print(f"\nLoading TMR from: {args.tmr_path} on device: {args.device}...")
    original_cwd = os.getcwd()
    os.chdir(str(TMR_ROOT))

    tmr_cfg = read_config(args.tmr_path)
    tmr = load_model_from_cfg(tmr_cfg, args.ckpt_name, eval_mode=True, device=args.device)
    text_model = instantiate(tmr_cfg.data.text_to_token_emb, device=args.device)
    motion_normalizer = instantiate(tmr_cfg.data.motion_loader.normalizer)

    os.chdir(original_cwd)

    if 'text' in raw:
        texts = raw['text']
    else:
        texts = [dataset.fixed_items[i % len(dataset)].get('text', '') for i in range(bs)]

    results_by_class = defaultdict(list)
    gt_motions_list = []

    for i in range(len(dataset)):
        item = dataset[i]
        gt_motion = item[0] if isinstance(item, (list, tuple)) else item['motion']
        gt_motions_list.append(gt_motion)

    print("Extracting latent embeddings with TMR...")
    with torch.inference_mode():
        text_x_dict = collate_x_dict(text_model(texts))
        text_x_dict = move_x_dict_to_device(text_x_dict, args.device)
        z_text = tmr.encode(text_x_dict, sample_mean=True).cpu().numpy()

        gen_x_dicts = []
        for i in range(bs):
            true_len = int(lengths[i])
            m = torch.from_numpy(guo_features[i]).float().squeeze(0)[:true_len]
            m = motion_normalizer(m)
            gen_x_dicts.append({"x": m, "length": true_len})
        gen_x_dict = collate_x_dict(gen_x_dicts)
        gen_x_dict = move_x_dict_to_device(gen_x_dict, args.device)
        z_gen = tmr.encode(gen_x_dict, sample_mean=True).cpu().numpy()

        gt_x_dicts = []
        for gt_motion in gt_motions_list:
            if torch.is_tensor(gt_motion):
                m = gt_motion.float()
            else:
                m = torch.from_numpy(gt_motion).float()
            m = motion_normalizer(m)
            gt_x_dicts.append({"x": m, "length": len(m)})
        gt_x_dict = collate_x_dict(gt_x_dicts)
        gt_x_dict = move_x_dict_to_device(gt_x_dict, args.device)
        z_gt = tmr.encode(gt_x_dict, sample_mean=True).cpu().numpy()

    # Count without cutoff over the entire sequence (including crop and SLERP
    # tail); do not exclude any time intervals.
    for i in range(bs):
        joints_i = motion[i].transpose(2, 0, 1)
        contact_i = compute_foot_contact_from_joints(joints_i, args.foot_detect_thres)
        length_i = int(lengths[i])
        expected_L = expected_classes[i]

        n_steps, _, _ = count_steps_in_generated_video(
            joints_i, contact_i, length_i,
            args.min_horizontal_m, args.min_height_m, args.min_straightness
        )
        results_by_class[expected_L].append(n_steps - expected_L)

    classes = sorted(results_by_class.keys())

    fid_score = calculate_fid(z_gen, z_gt)
    tmr_score = calculate_tmr_matching_score(z_text, z_gen)

    # Compute Guo metrics (matching score and FID on 512-dimensional embeddings)
    print("\nExtracting embeddings with the Guo evaluator...")
    gen_guo = [guo_features[i].squeeze(0)[:int(lengths[i])].astype(np.float32)
               for i in range(bs)]
    gt_guo = [(m.numpy() if torch.is_tensor(m) else np.asarray(m)).astype(np.float32)
              for m in gt_motions_list]

    guo_results = evaluate_guo(
        gen_guo, gt_guo, list(texts), evaluator=load_guo_evaluator(args.device),
        tokens=load_guo_tokens(texts), batch_size=args.guo_batch_size,
        matching_pool=guo_pool, seed=args.guo_seed,
    )

    print("\n" + "=" * 80)
    print("[NO-CUTOFF VERSION - raw count over the entire sequence]")
    print(f"{'Class':<8} {'N videos':<9} {'Exact':<10} {'±1 step':<11} "
          f"{'Error >=2':<12} {'Exact+/-1 (cumulative)'}")
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
    print(f"{'TOTAL':<8} {tot_n:<9} "
          f"{100.0*tot_exact/tot_n:>6.1f}%   "
          f"{100.0*tot_off1/tot_n:>7.1f}%   "
          f"{100.0*tot_off_more/tot_n:>8.1f}%    "
          f"{100.0*(tot_exact+tot_off1)/tot_n:>6.1f}%")

    print("\n" + "=" * 62)
    print("                    QUALITY METRICS                           ")
    print("=" * 62)
    print(f" TMR Matching Score (Cosine similarity, higher is better): {tmr_score:.4f}")
    print(f" TMR-FID (Fréchet distance, lower is better)             : {fid_score:.4f}")
    print("-" * 62)
    print(f" Guo Matching Score (Euclidean distance, lower is better) : "
          f"{guo_results['matching_score']:.4f}")
    print(f" Guo-FID (lower is better)                                 : "
          f"{guo_results['fid']:.4f}")
    print(f" Guo clips: {guo_results['n_generated']} generated, "
          f"{guo_results['n_reference']} reference, "
          f"{guo_results['n_matching']} used for matching "
          f"(pool={guo_pool})")
    print("=" * 62)


if __name__ == '__main__':
    main()