#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script autonomo per visualizzare motion in formato "guofeats" (HumanML3D,
263 features) oppure in formato "batch" (es. output di script tipo MDM
sample.py, un .npy contenente un dict con piu' motion/video insieme) come
video MP4 o GIF.

Non dipende da nessun altro modulo del progetto. Servono le librerie di
terze parti: numpy, matplotlib (e torch SOLO se il file di input e' nel
formato guofeats a 263 feature; per il formato "batch" torch non serve).
Per il salvataggio in MP4 serve anche ffmpeg installato sul sistema; in
mancanza, lo script ripiega automaticamente sulla GIF.

Formati di input supportati
----------------------------
1) "guofeats singolo": array .npy di forma (T, 263) oppure (1, T, 263).
   Un solo video, feature HumanML3D grezze (263 numeri per frame).

2) "batch multi-video": file .npy che contiene un OGGETTO (es. dict, tipo
   quelli salvati con np.save(..., allow_pickle=True) da script come
   MDM sample.py) con almeno la chiave:
       - "motion": array (N, njoints, 3, T)  -> N video impacchettati
   e opzionalmente:
       - "text": lista di N descrizioni testuali (una per video)
       - "lengths": array di N interi, frame validi per ciascun video
                    (utile se il tensore e' stato paddato a T frame fissi)
       - "num_samples", "num_repetitions": interi, usati solo per dare un
         nome piu' leggibile ai file di output (sampleXX_repYY.mp4)
   In questo caso lo script estrae e renderizza OGNI video al suo interno,
   uno per file di output.

Esempi d'uso
------------
    # Caso 1: singolo motion guofeats (comportamento originale)
    python render_guofeats_multi.py motion.npy

    # Caso 2: file batch con N video -> renderizza tutti in una cartella
    python render_guofeats_multi.py results.npy --outdir video_output

    # Renderizzare solo il video con indice 3 (0-based) del batch
    python render_guofeats_multi.py results.npy --index 3 -o sample.mp4

    # GIF esplicita, 20 fps
    python render_guofeats_multi.py results.npy --outdir out --fps 20 --format gif
"""

import os
import re
import argparse
import numpy as np


# =============================================================================
#  Funzioni quaternioni (estratte da common/quaternion.py)
#  NB: importano torch "pigramente" (solo se e la pipeline guofeats e'
#  effettivamente usata), cosi' il formato "batch" funziona anche senza torch.
# =============================================================================
def qinv(q):
    """Inverso (coniugato, per quaternioni unitari) di q. Shape (*, 4)."""
    assert q.shape[-1] == 4, "q deve avere shape (*, 4)"
    mask = q.new_ones(q.shape)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    """
    Ruota il vettore (o i vettori) v secondo il quaternione q.
    q: shape (*, 4), v: shape (*, 3) -> ritorna shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    import torch

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


# =============================================================================
#  Conversione guofeats (263) -> joints 3D (T, 22, 3)
#  (estratta da guofeats/motion_representation.py)
# =============================================================================
def recover_root_rot_pos(data):
    """Recupera rotazione e posizione del root dalle feature guofeats."""
    import torch

    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Rotazione attorno all'asse Y a partire dalla velocita' angolare
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # Applica la rotazione attorno a Y alla posizione del root
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num=22):
    """Ricostruisce le posizioni dei joint dalla parte 'RIC' delle feature."""
    import torch

    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Aggiunge la rotazione attorno a Y ai joint locali
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )

    # Aggiunge la posizione XZ del root ai joint
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatena root e joint
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def axes_to_render_convention(x, horiz2, vertical):
    """
    Riordina 3 componenti (x, secondo_orizzontale, verticale) nella
    convenzione usata da render_animation: (x, y, z) con z = asse di
    gravita'/verticale. Corrisponde esattamente alla trasformazione usata
    in origine da guofeats_to_joints.
    """
    import torch

    return torch.stack((x, -horiz2, vertical), axis=-1)


def guofeats_to_joints(data):
    """
    Converte feature guofeats (T, 263) in joints 3D (T, 22, 3),
    con Z come asse verticale (gravita').
    """
    import torch

    joints_num = 22
    joints = recover_from_ric(data, joints_num)
    # joints e' in convenzione HumanML3D: asse0=X, asse1=Y(verticale), asse2=Z(profondita')
    x, vertical, horiz2 = torch.unbind(joints, axis=-1)
    return axes_to_render_convention(x, horiz2, vertical)


def raw_joints_to_render_joints(joints):
    """
    Come guofeats_to_joints, ma parte gia' da joint 3D grezzi (numpy),
    nella stessa convenzione HumanML3D (asse0=X, asse1=Y verticale,
    asse2=Z profondita'), SENZA bisogno di torch/quaternioni: qui i joint
    sono gia' stati calcolati a monte (es. da uno script di generazione
    tipo MDM), serve solo riordinare gli assi per il renderer.

    Input:  joints numpy array, shape (..., 3) con convenzione (X, Y_up, Z_depth)
    Output: numpy array, shape (..., 3) con convenzione (X, -Z_depth, Y_up)
            pronta per render_animation (asse2 = verticale).
    """
    joints = np.asarray(joints, dtype=np.float32)
    x = joints[..., 0]
    vertical = joints[..., 1]
    horiz2 = joints[..., 2]
    return np.stack((x, -horiz2, vertical), axis=-1)


# =============================================================================
#  Rendering con matplotlib (estratto da renderer/matplotlib.py)
# =============================================================================
COLORS = ("black", "magenta", "red", "green", "blue")

# Albero cinematico per 22 joint (HumanML3D / guoh3djoints), senza mani
KINEMATIC_TREE = [
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
]


def init_axis(fig, title, radius=1.5):
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=20.0, azim=-60)

    fact = 2
    ax.set_xlim3d([-radius / fact, radius / fact])
    ax.set_ylim3d([-radius / fact, radius / fact])
    ax.set_zlim3d([0, radius])

    ax.set_aspect("auto")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()
    ax.grid(False)

    ax.set_title(title, loc="center", wrap=True)
    return ax


def plot_floor(ax, minx, maxx, miny, maxy, minz):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Piano XZ
    verts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz],
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.add_collection3d(xz_plane)

    # Piano XZ piu' grande
    radius = max((maxx - minx), (maxy - miny))
    minx_all = (maxx + minx) / 2 - radius
    maxx_all = (maxx + minx) / 2 + radius
    miny_all = (maxy + miny) / 2 - radius
    maxy_all = (maxy + miny) / 2 + radius

    verts = [
        [minx_all, miny_all, minz],
        [minx_all, maxy_all, minz],
        [maxx_all, maxy_all, minz],
        [maxx_all, miny_all, minz],
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)
    return ax


def update_camera(ax, root, radius=1.5):
    fact = 2
    ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
    ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])


def render_animation(
    joints: np.ndarray,
    output: str,
    title: str = "",
    fps: float = 20.0,
    colors=COLORS,
    figsize=(4, 4),
    fontsize: int = 15,
):
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as pe

    kinematic_tree = KINEMATIC_TREE

    # Euristica per la dimensione del font in base alla lunghezza del titolo
    fontsize = fontsize - (len(title) - 30) / 20
    plt.rcParams.update({"font.size": fontsize})

    # Z e' l'asse di gravita'
    x, y, z = 0, 1, 2
    joints = joints.copy()

    fig = plt.figure(figsize=figsize)
    ax = init_axis(fig, title)

    # Traiettoria del root sul piano XY
    trajectory = joints[:, 0, [x, y]]
    avg_segment_length = (
        np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    )
    draw_offset = int(25 / avg_segment_length)
    (spline_line,) = ax.plot(*trajectory.T, zorder=10, color="white")

    # Pavimento
    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax, minx, maxx, miny, maxy, 0)

    # Mette il personaggio sul pavimento
    height_offset = np.min(joints[:, :, z])
    joints = joints.copy()
    joints[:, :, z] -= height_offset

    lines = []
    initialized = False

    def update(frame):
        nonlocal initialized
        skeleton = joints[frame]

        root = skeleton[0]
        update_camera(ax, root)

        hcolors = colors
        for index, (chain, color) in enumerate(
            zip(reversed(kinematic_tree), reversed(hcolors))
        ):
            if not initialized:
                lines.append(
                    ax.plot(
                        skeleton[chain, x],
                        skeleton[chain, y],
                        skeleton[chain, z],
                        linewidth=6.0,
                        color=color,
                        zorder=20,
                        path_effects=[pe.SimpleLineShadow(), pe.Normal()],
                    )
                )
            else:
                lines[index][0].set_xdata(skeleton[chain, x])
                lines[index][0].set_ydata(skeleton[chain, y])
                lines[index][0].set_3d_properties(skeleton[chain, z])
                lines[index][0].set_color(color)

        left = max(frame - draw_offset, 0)
        right = min(frame + draw_offset, trajectory.shape[0])

        spline_line.set_xdata(trajectory[left:right, 0])
        spline_line.set_ydata(trajectory[left:right, 1])
        spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
        initialized = True

    fig.tight_layout()
    frames = joints.shape[0]
    anim = FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, repeat=False
    )

    ext = os.path.splitext(output)[1].lower()
    try:
        if ext == ".gif":
            from matplotlib.animation import PillowWriter

            anim.save(output, writer=PillowWriter(fps=int(round(fps))))
        else:
            anim.save(output, fps=fps)  # usa ffmpeg per l'MP4
        print(f"Salvato: {output}")
    except Exception as e:
        # Ripiego sulla GIF se l'MP4 (ffmpeg) non e' disponibile
        print(f"Salvataggio in {ext or 'mp4'} fallito ({e}); provo con la GIF...")
        from matplotlib.animation import PillowWriter

        gif_output = os.path.splitext(output)[0] + ".gif"
        anim.save(gif_output, writer=PillowWriter(fps=int(round(fps))))
        print(f"Salvato: {gif_output}")

    plt.close()


# =============================================================================
#  Utility
# =============================================================================
def _load_stat(path):
    """Carica un vettore di statistiche (mean/std) da .npy o .pt."""
    if path.endswith(".pt"):
        import torch

        v = torch.load(path, map_location="cpu")
        if hasattr(v, "float"):
            v = v.float().numpy()
        return np.asarray(v)
    return np.load(path)


def _sanitize_filename(text, max_len=40):
    """Trasforma una stringa di testo in un nome file leggibile e sicuro."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "motion"
    return text[:max_len]


def detect_input_kind(path):
    """
    Ispeziona il file .npy e determina se contiene:
      - 'guofeats': un singolo motion (T, 263) / (1, T, 263)
      - 'batch': un oggetto (dict) con piu' video impacchettati, chiave 'motion'
    Ritorna una tupla (kind, payload).
    """
    raw = np.load(path, allow_pickle=True)

    # Un oggetto python (tipicamente un dict) salvato con allow_pickle=True
    # viene caricato come array 0-dim di dtype=object.
    if raw.dtype == object and raw.shape == ():
        obj = raw.item()
        if isinstance(obj, dict) and "motion" in obj:
            return "batch", obj
        raise ValueError(
            "Il file .npy contiene un oggetto pickled ma non un dict con "
            "chiave 'motion'; formato non riconosciuto."
        )

    arr = raw.squeeze()
    return "guofeats", arr


def render_single_guofeats(motions, output, args):
    """Pipeline originale: un solo video in formato guofeats (263 feature)."""
    import torch

    motions = np.asarray(motions, dtype=np.float32)
    if motions.ndim != 2 or motions.shape[-1] != 263:
        raise ValueError(
            f"Atteso un array (T, 263), ricevuto shape {motions.shape}. "
            "Questo script gestisce solo il formato guofeats a 263 feature "
            "per il caso 'singolo motion'."
        )

    if args.mean and args.std:
        mean = _load_stat(args.mean).astype(np.float32)
        std = _load_stat(args.std).astype(np.float32)
        print("De-normalizzazione delle feature guofeats (motions * std + mean).")
        motions = motions * std + mean

    joints = guofeats_to_joints(torch.from_numpy(motions)).numpy()

    print(f"Frame: {joints.shape[0]} | joints: {joints.shape[1]} | fps: {args.fps}")
    render_animation(joints, output, title=args.title, fps=args.fps)


def render_batch(obj, args):
    """
    Pipeline per file .npy 'batch' con N video impacchettati insieme
    (es. output di script tipo MDM sample.py):
        obj["motion"]  -> (N, njoints, 3, T)
        obj["text"]    -> lista di N stringhe (opzionale)
        obj["lengths"] -> array di N interi (opzionale)
        obj["num_samples"], obj["num_repetitions"] -> interi (opzionali)
    """
    motion = np.asarray(obj["motion"], dtype=np.float32)
    if motion.ndim != 4 or motion.shape[2] != 3:
        raise ValueError(
            f"Atteso obj['motion'] di forma (N, njoints, 3, T), ricevuto "
            f"shape {motion.shape}."
        )

    n_videos, n_joints, _, t_frames = motion.shape
    texts = obj.get("text", [""] * n_videos)
    lengths = obj.get("lengths", [t_frames] * n_videos)
    num_samples = obj.get("num_samples", n_videos)
    num_repetitions = obj.get("num_repetitions", max(1, n_videos // max(1, num_samples)))

    print(
        f"Trovati {n_videos} video nel file batch "
        f"({num_samples} sample x {num_repetitions} ripetizioni), "
        f"{n_joints} joint, fino a {t_frames} frame ciascuno."
    )

    # Quali indici renderizzare
    if args.index is not None:
        if not (0 <= args.index < n_videos):
            raise ValueError(f"--index {args.index} fuori range (0..{n_videos - 1})")
        indices = [args.index]
    else:
        indices = list(range(n_videos))

    # Cartella/prefisso di output
    ext = ".gif" if args.format == "gif" else ".mp4"
    if args.output:
        # Se e' stato chiesto un solo video con -o, usalo come file esatto.
        if len(indices) == 1:
            outdir = os.path.dirname(args.output) or "."
            os.makedirs(outdir, exist_ok=True)
            out_paths = {indices[0]: args.output}
        else:
            outdir = args.output
            os.makedirs(outdir, exist_ok=True)
            out_paths = None
    else:
        outdir = args.outdir or "video_output"
        os.makedirs(outdir, exist_ok=True)
        out_paths = None

    for idx in indices:
        sample_i = idx // num_repetitions
        rep_i = idx % num_repetitions
        length = int(lengths[idx]) if lengths is not None else t_frames
        length = max(1, min(length, t_frames))

        arr = motion[idx][:, :, :length]          # (njoints, 3, length)
        arr = np.transpose(arr, (2, 0, 1))         # (length, njoints, 3)
        joints = raw_joints_to_render_joints(arr)  # convenzione per render_animation

        text_i = texts[idx] if idx < len(texts) else ""
        title = args.title if args.title else text_i

        if out_paths is not None:
            out_path = out_paths[idx]
        else:
            base = f"sample{sample_i:02d}_rep{rep_i:02d}"
            if text_i:
                base += f"_{_sanitize_filename(text_i)}"
            out_path = os.path.join(outdir, base + ext)

        print(
            f"[{idx + 1}/{n_videos}] sample={sample_i} rep={rep_i} "
            f"frame={length} -> {out_path}  (\"{text_i}\")"
        )
        render_animation(joints, out_path, title=title, fps=args.fps)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualizza motion guofeats (263 feature) o file .npy 'batch' "
            "con piu' video impacchettati (es. output MDM sample.py) come "
            "MP4 o GIF."
        )
    )
    parser.add_argument(
        "input", help="File .npy di input: (T, 263)/(1, T, 263), oppure dict batch con chiave 'motion'."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "File (singolo motion o --index) o cartella (batch, piu' video) di output. "
            "Default: <nome_input>.mp4 per il caso singolo, cartella 'video_output/' per il batch."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Cartella di output per il caso batch (alternativa a -o). Default: 'video_output'.",
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Formato di output per il caso batch (default mp4; ripiega su gif se ffmpeg manca).",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Frame al secondo (default 20).")
    parser.add_argument(
        "--title",
        default="",
        help="Titolo opzionale del video. Nel caso batch, se omesso, usa il testo associato a ciascun video.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="(Solo caso batch) Renderizza solo il video con questo indice 0-based, invece di tutti.",
    )
    parser.add_argument(
        "--mean",
        default=None,
        help="(Solo caso guofeats singolo) file mean .npy/.pt per de-normalizzare le feature.",
    )
    parser.add_argument(
        "--std",
        default=None,
        help="(Solo caso guofeats singolo) file std .npy/.pt per de-normalizzare le feature.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="(Opzionale) percorso dell'eseguibile ffmpeg per l'MP4.",
    )
    args = parser.parse_args()

    if args.ffmpeg_path:
        import matplotlib

        matplotlib.rcParams["animation.ffmpeg_path"] = args.ffmpeg_path

    kind, payload = detect_input_kind(args.input)

    if kind == "guofeats":
        output = args.output or (
            os.path.splitext(os.path.basename(args.input))[0] + ".mp4"
        )
        render_single_guofeats(payload, output, args)
    else:  # "batch"
        render_batch(payload, args)


if __name__ == "__main__":
    main()