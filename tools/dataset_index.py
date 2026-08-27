"""
Costruisce l'indice di un dataset "a classi di passi" (dataset_ris / dataset_risv2)
e ne calcola le statistiche.

Struttura attesa:
    <root>/<N>_step/<nome>.npy   feature HumanML3D vec263 NON normalizzate, 20 fps
    <root>/<N>_step/<nome>.txt   caption testuale
    <root>/<N>_step/<nome>.mp4   render (opzionale)
    <root>/{train,val,test}.txt  una riga per file .npy (path relativo alla repo)

Usato sia da dataset_stats.py sia da dataset_viewer.py.
"""
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

FPS = 20  # HumanML3D


def root_trajectory(data):
    """Ricostruisce la traiettoria del root (pelvis) sul piano di terra a
    partire dalle feature vec263, integrando velocita' angolare e lineare.

    E' la trasposizione in numpy di recover_root_rot_pos() di
    data_loaders/humanml/scripts/motion_process.py: e' la stessa curva che i
    render disegnano come linea bianca a terra.

    data: (T, 263) NON normalizzato. Ritorna (T, 2) in metri, ordine (x, z).
    """
    data = np.asarray(data, dtype=np.float64)
    # [0] = velocita' di rotazione attorno a Y, [1:3] = velocita' lineare XZ
    # espressa nel sistema di riferimento locale (solidale al busto)
    ang = np.zeros(len(data))
    ang[1:] = np.cumsum(data[:-1, 0])

    vx = np.zeros(len(data))
    vz = np.zeros(len(data))
    vx[1:], vz[1:] = data[:-1, 1], data[:-1, 2]

    # porta gli spostamenti dal frame locale a quello globale (rotazione -2*ang
    # attorno a Y, come fa qrot(qinv(q), .) con q = (cos ang, 0, sin ang, 0))
    a = 2.0 * ang
    ca, sa = np.cos(a), np.sin(a)
    gx = ca * vx - sa * vz
    gz = sa * vx + ca * vz

    return np.stack([np.cumsum(gx), np.cumsum(gz)], axis=1)


# Layout HumanML3D a 22 joint (motion_process.py:462): caviglia/punta sinistra
# e destra. I 4 canali di contatto in coda alle feature seguono lo stesso ordine
# (motion_process.py:168 -> [feet_l, feet_r]).
FID_L, FID_R = [7, 10], [8, 11]
CONTACT_SLICE = slice(259, 263)


def recover_joints(data, joints_num=22):
    """Posizioni globali dei joint (T, 22, 3) dalle feature vec263.

    Trasposizione in numpy di recover_from_ric(): riporta le posizioni
    relative al root (ric_data, dim 4:67) nel sistema globale e vi somma la
    traiettoria del root. data va passato NON normalizzato.
    """
    data = np.asarray(data, dtype=np.float64)
    ang = np.zeros(len(data))
    ang[1:] = np.cumsum(data[:-1, 0])
    a = 2.0 * ang
    ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]

    pos = data[:, 4:4 + (joints_num - 1) * 3].reshape(len(data), joints_num - 1, 3)
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    gx, gz = ca * x - sa * z, sa * x + ca * z

    traj = root_trajectory(data)
    root = np.stack([traj[:, 0], data[:, 3], traj[:, 1]], axis=1)

    joints = np.empty((len(data), joints_num, 3))
    joints[:, 0] = root
    joints[:, 1:, 0] = gx + traj[:, :1]
    joints[:, 1:, 1] = y
    joints[:, 1:, 2] = gz + traj[:, 1:2]
    return joints


def foot_travel(data, joints_num=22):
    """Metri percorsi dai PIEDI sul piano di terra (assi X e Z; in guofeats la
    verticale e' Y, e pesa solo il 3-4% della norma 3D, quindi la scartiamo).

    Calcola due varianti dello spostamento della caviglia:

    foot_m      somma solo le fasi di volo. Un piede e' appoggiato quando
                almeno uno dei suoi due canali di contatto (tallone o punta)
                e' attivo; un segmento t -> t+1 conta se il piede e' staccato
                ad ALMENO uno dei due estremi, cosi' da includere anche lo
                stacco e l'atterraggio, in cui il piede si muove eccome.
    foot_all_m  somma TUTTI i frame, senza guardare i bit di contatto.

    La differenza fra le due (skate_m) e' lo spostamento della caviglia mentre
    il piede risulta appoggiato: e' scivolamento, cioe' foot skating.

    A differenza della traiettoria del pelvis nessuna delle due si annulla
    quando il soggetto passeggia sul posto o torna al punto di partenza.
    """
    joints = recover_joints(data, joints_num)
    contact = data[:, CONTACT_SLICE] > 0.5

    gated = ungated = 0.0
    per_foot = []
    for ankle, (c0, c1) in ((FID_L[0], (0, 1)), (FID_R[0], (2, 3))):
        planted = contact[:, c0] | contact[:, c1]
        d = np.linalg.norm(np.diff(joints[:, ankle], axis=0)[:, [0, 2]], axis=1)
        flying = ~planted[:-1] | ~planted[1:]
        per_foot.append(float(d[flying].sum()))
        gated += per_foot[-1]
        ungated += float(d.sum())

    n = len(data)
    airborne = float((~(contact[:, 0] | contact[:, 1])).mean() +
                     (~(contact[:, 2] | contact[:, 3])).mean()) / 2
    return {
        'foot_m': round(gated, 3),
        'foot_all_m': round(ungated, 3),
        'skate_m': round(ungated - gated, 3),
        'foot_l_m': round(per_foot[0], 3),
        'foot_r_m': round(per_foot[1], 3),
        'airborne_frac': round(airborne, 3),
        'foot_speed_ms': round(gated / (n / FPS), 3) if n else 0.0,
    }


def travel_metrics(data, fps=FPS):
    """Metri percorsi dal pelvis: lunghezza del percorso, spostamento netto
    inizio->fine, velocita' media e rettilineita' del tragitto."""
    traj = root_trajectory(data)
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    path = float(steps.sum())
    disp = float(np.linalg.norm(traj[-1] - traj[0])) if len(traj) > 1 else 0.0
    duration = len(data) / fps
    out = {
        'path_m': round(path, 3),
        'disp_m': round(disp, 3),
        'speed_ms': round(path / duration, 3) if duration else 0.0,
        # 1.0 = tragitto perfettamente rettilineo, ~0 = giro chiuso / sul posto
        'straightness': round(disp / path, 3) if path > 1e-6 else 0.0,
    }
    out.update(foot_travel(data))
    return out


def class_steps(cls):
    """Numero di passi di una classe, o None se non e' quantificata
    (es. 'few_steps' / 'many_steps' nel dataset di partenza)."""
    m = re.match(r'^(\d+)_steps?$', cls)
    return int(m.group(1)) if m else None


def find_step_dirs(root):
    """Cartelle-classe: '<n>_step' ordinate per numero, poi le classi non
    numeriche ('few_steps', 'many_steps') in coda in ordine alfabetico."""
    out = []
    for c in glob.glob(os.path.join(root, '*_step')) + glob.glob(os.path.join(root, '*_steps')):
        if not os.path.isdir(c):
            continue
        base = os.path.basename(c)
        n = class_steps(base)
        out.append((0 if n is not None else 1, n if n is not None else 0, base, c))
    out.sort()
    return [c for _, _, _, c in out]


def source_id(npy_path):
    """Risale al clip HumanML3D di origine: toglie il prefisso cut_<n>_,
    i suffissi _gen<k>_<j> / _aug<k>_<j> e la M del mirroring."""
    b = os.path.splitext(os.path.basename(npy_path))[0]
    b = re.sub(r'^cut_\d+_', '', b)
    b = re.sub(r'_(gen|aug)\d+_\d+$', '', b)
    return b.lstrip('M')


def origin_kind(npy_path):
    b = os.path.splitext(os.path.basename(npy_path))[0]
    if re.search(r'_gen\d+_\d+$', b):
        return 'gen'
    if re.search(r'_aug\d+_\d+$', b):
        return 'aug'
    if b.startswith('cut_'):
        return 'cut'
    return 'orig'


def read_splits(root):
    """Ritorna {path_npy_normalizzato: 'train'|'val'|'test'}."""
    assign = {}
    for split in ('train', 'val', 'test'):
        p = os.path.join(root, f'{split}.txt')
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    assign[os.path.normpath(line)] = split
    return assign


def build_index(root, with_hash=True):
    """Ritorna (samples, meta). samples e' una lista di dict, uno per .npy."""
    root = os.path.normpath(root)
    splits = read_splits(root)
    samples = []

    for class_dir in find_step_dirs(root):
        cls = os.path.basename(class_dir)
        for npy in sorted(glob.glob(os.path.join(class_dir, '*.npy'))):
            base = os.path.splitext(npy)[0]
            txt, mp4 = base + '.txt', base + '.mp4'

            arr = np.load(npy)
            n_frames = int(arr.shape[0])
            n_feats = int(arr.shape[1]) if arr.ndim > 1 else 0
            travel = travel_metrics(arr) if n_feats >= 4 and n_frames > 1 else {
                'path_m': 0.0, 'disp_m': 0.0, 'speed_ms': 0.0, 'straightness': 0.0}

            caption = ''
            if os.path.exists(txt):
                with open(txt, encoding='utf-8', errors='replace') as f:
                    caption = f.read().strip()

            samples.append({
                'id': len(samples),
                'name': os.path.basename(base),
                'cls': cls,
                'npy': os.path.relpath(npy, root),
                'mp4': os.path.relpath(mp4, root) if os.path.exists(mp4) else None,
                'caption': caption,
                'frames': n_frames,
                'feats': n_feats,
                'seconds': round(n_frames / FPS, 2),
                'split': splits.get(os.path.normpath(npy), '?'),
                'origin': origin_kind(npy),
                'source': source_id(npy),
                'hash': hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
                        if with_hash else None,
                **travel,
            })

    meta = {'root': root, 'fps': FPS, 'n_samples': len(samples),
            'classes': sorted({s['cls'] for s in samples},
                              key=lambda c: (class_steps(c) is None, class_steps(c) or 0, c))}

    if with_hash:
        _annotate_duplicates_and_leakage(samples)
    return samples, meta


def _annotate_duplicates_and_leakage(samples):
    """Marca i cloni byte-identici e le fughe di dati tra split."""
    by_hash = defaultdict(list)
    for s in samples:
        by_hash[s['hash']].append(s)

    train_hashes = {s['hash'] for s in samples if s['split'] == 'train'}
    train_sources = {s['source'] for s in samples if s['split'] == 'train'}

    for s in samples:
        group = by_hash[s['hash']]
        s['dup_count'] = len(group)
        s['dup_with'] = [o['name'] for o in group if o is not s][:8]
        # fuga: un campione di val/test identico (o dallo stesso clip sorgente)
        # a qualcosa gia' visto in training
        s['leak'] = None
        if s['split'] in ('val', 'test'):
            if s['hash'] in train_hashes:
                s['leak'] = 'exact'
            elif s['source'] in train_sources:
                s['leak'] = 'source'


def compute_stats(samples, meta):
    """Statistiche aggregate globali e per classe."""
    fps = meta['fps']

    def block(subset):
        if not subset:
            return None
        lens = np.array([s['frames'] for s in subset], dtype=float)
        path = np.array([s['path_m'] for s in subset], dtype=float)
        disp = np.array([s['disp_m'] for s in subset], dtype=float)
        speed = np.array([s['speed_ms'] for s in subset], dtype=float)
        foot = np.array([s['foot_m'] for s in subset], dtype=float)
        foot_all = np.array([s['foot_all_m'] for s in subset], dtype=float)
        skate = np.array([s['skate_m'] for s in subset], dtype=float)
        air = np.array([s['airborne_frac'] for s in subset], dtype=float)
        caps = Counter(s['caption'] for s in subset)
        return {
            'n': len(subset),
            'unique_motions': len({s['hash'] for s in subset}),
            'unique_sources': len({s['source'] for s in subset}),
            'unique_captions': len(caps),
            'top_captions': caps.most_common(5),
            'frames_mean': float(lens.mean()),
            'frames_median': float(np.median(lens)),
            'frames_min': int(lens.min()),
            'frames_max': int(lens.max()),
            'sec_mean': float(lens.mean() / fps),
            'sec_median': float(np.median(lens) / fps),
            'sec_min': float(lens.min() / fps),
            'sec_max': float(lens.max() / fps),
            'sec_total': float(lens.sum() / fps),
            'path_mean': float(path.mean()),
            'path_median': float(np.median(path)),
            'path_min': float(path.min()),
            'path_max': float(path.max()),
            'path_std': float(path.std()),
            'path_total': float(path.sum()),
            'disp_mean': float(disp.mean()),
            'speed_mean': float(speed.mean()),
            'foot_mean': float(foot.mean()),
            'foot_median': float(np.median(foot)),
            'foot_min': float(foot.min()),
            'foot_max': float(foot.max()),
            'foot_std': float(foot.std()),
            'foot_total': float(foot.sum()),
            'foot_all_mean': float(foot_all.mean()),
            'foot_all_median': float(np.median(foot_all)),
            'skate_mean': float(skate.mean()),
            'airborne_mean': float(air.mean()),
            'origins': dict(Counter(s['origin'] for s in subset)),
            'splits': dict(Counter(s['split'] for s in subset)),
        }

    stats = {
        'meta': meta,
        'overall': block(samples),
        'per_class': {c: block([s for s in samples if s['cls'] == c])
                      for c in meta['classes']},
        'per_split': {sp: block([s for s in samples if s['split'] == sp])
                      for sp in ('train', 'val', 'test')
                      if any(s['split'] == sp for s in samples)},
    }

    leaks = [s for s in samples if s.get('leak')]
    stats['leakage'] = {
        'exact': sum(1 for s in leaks if s['leak'] == 'exact'),
        'same_source': sum(1 for s in leaks if s['leak'] == 'source'),
        'eval_total': sum(1 for s in samples if s['split'] in ('val', 'test')),
    }
    stats['duplicates'] = {
        'redundant_files': len(samples) - len({s['hash'] for s in samples}),
        'groups': sum(1 for _, g in Counter(s['hash'] for s in samples).items() if g > 1),
    }
    return stats


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'dataset_ris'
    samples, meta = build_index(root)
    print(json.dumps(compute_stats(samples, meta), indent=2, default=str))
