#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script autonomo per visualizzare un motion nel formato "guofeats" (HumanML3D,
263 features) come video MP4 o GIF.

Non dipende da nessun altro modulo del progetto: tutto il codice necessario
(conversione guofeats -> joints 3D e rendering con matplotlib) e' contenuto qui.
Servono solo le librerie di terze parti: numpy, torch, matplotlib
(per il salvataggio in MP4 serve anche ffmpeg installato sul sistema; in
mancanza, lo script ripiega automaticamente sulla GIF).

Esempi d'uso
------------
    # MP4 (default), stesso nome del file di input
    python render_guofeats.py motion.npy

    # GIF esplicita, 20 fps
    python render_guofeats.py motion.npy -o motion.gif --fps 20

    # con de-normalizzazione (se le feature sono standardizzate)
    python render_guofeats.py motion.npy --mean mean.pt --std std.pt

Formato di input
----------------
File .npy contenente un array di forma (T, 263) oppure (1, T, 263).
T = numero di frame. 263 = dimensione delle feature HumanML3D (guofeats).
"""

import os
import argparse
import numpy as np
import torch


# =============================================================================
#  Funzioni quaternioni (estratte da common/quaternion.py)
# =============================================================================
def qinv(q):
    """Inverso (coniugato, per quaternioni unitari) di q. Shape (*, 4)."""
    assert q.shape[-1] == 4, "q deve avere shape (*, 4)"
    mask = torch.ones_like(q)
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


def guofeats_to_joints(data):
    """
    Converte feature guofeats (T, 263) in joints 3D (T, 22, 3),
    con Z come asse verticale (gravita').
    """
    joints_num = 22
    joints = recover_from_ric(data, joints_num)
    # Rimette Z come terza coordinata: scambia Y e Z e nega Y
    x, z, my = torch.unbind(joints, axis=-1)
    return torch.stack((x, -my, z), axis=-1)


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
        v = torch.load(path, map_location="cpu")
        if isinstance(v, torch.Tensor):
            v = v.float().numpy()
        return np.asarray(v)
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza un motion guofeats (263 feature) come MP4 o GIF."
    )
    parser.add_argument("input", help="File .npy di input (T, 263) o (1, T, 263).")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="File di output (.mp4 o .gif). Default: <nome_input>.mp4 nella cartella corrente.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Frame al secondo (default 20).")
    parser.add_argument("--title", default="", help="Titolo opzionale del video.")
    parser.add_argument(
        "--mean",
        default=None,
        help="(Opzionale) file mean .npy/.pt per de-normalizzare le feature.",
    )
    parser.add_argument(
        "--std",
        default=None,
        help="(Opzionale) file std .npy/.pt per de-normalizzare le feature.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=None,
        help="(Opzionale) percorso dell'eseguibile ffmpeg per l'MP4.",
    )
    args = parser.parse_args()

    # Output di default: stesso nome del file di input ma con estensione .mp4,
    # creato nella cartella corrente (quella da cui si lancia lo script).
    output = args.output or (
        os.path.splitext(os.path.basename(args.input))[0] + ".mp4"
    )

    if args.ffmpeg_path:
        import matplotlib

        matplotlib.rcParams["animation.ffmpeg_path"] = args.ffmpeg_path

    # Carica e normalizza la forma a (T, 263)
    motions = np.load(args.input, allow_pickle=True).squeeze()
    motions = np.asarray(motions, dtype=np.float32)
    if motions.ndim != 2 or motions.shape[-1] != 263:
        raise ValueError(
            f"Atteso un array (T, 263), ricevuto shape {motions.shape}. "
            "Questo script gestisce solo il formato guofeats a 263 feature."
        )

    # De-normalizzazione opzionale
    if args.mean and args.std:
        mean = _load_stat(args.mean).astype(np.float32)
        std = _load_stat(args.std).astype(np.float32)
        print("De-normalizzazione delle feature guofeats (motions * std + mean).")
        motions = motions * std + mean

    # guofeats (263) -> joints 3D (T, 22, 3)
    joints = guofeats_to_joints(torch.from_numpy(motions)).numpy()

    print(f"Frame: {joints.shape[0]} | joints: {joints.shape[1]} | fps: {args.fps}")
    render_animation(joints, output, title=args.title, fps=args.fps)


if __name__ == "__main__":
    main()