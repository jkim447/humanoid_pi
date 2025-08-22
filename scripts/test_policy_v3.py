"""
USAGE:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_policy_egodex.py \
  --root /iris/projects/humanoid/dataset/ego_dex \
  --part part2 \
  --task furniture_bench   \
  --ckpt checkpoints/pi0_galaxea/my_egodex_experiment/10000
"""

# ───────── imports ─────────
import os, glob, random, argparse
import cv2, h5py, numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.shared import download

# ───────── helpers: SE(3) & state encoding ─────────
def _inv(T: np.ndarray) -> np.ndarray:
    Rm, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = Rm.T
    inv[:3, 3]  = -Rm.T @ t
    return inv

def _pose_world_to_cam(T_world_obj: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    return _inv(T_world_cam) @ T_world_obj

def _rotmat_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    # [r00, r10, r20, r01, r11, r21]
    return np.array([Rm[0,0], Rm[1,0], Rm[2,0], Rm[0,1], Rm[1,1], Rm[2,1]], dtype=np.float32)

def make_pi0_state_vec(f: h5py.File, t: int) -> np.ndarray:
    """
    32-D layout: [L_xyz(3), L_rot6d(6), 0.0, R_xyz(3), R_rot6d(6), 0.0, zeros(12)]
    Camera-frame wrists; no hand joints in EgoDex -> last 12 zeros.
    """
    T_world_cam = f["transforms"]["camera"][t]  # camera pose in world
    L_world     = f["transforms"]["leftHand"][t]
    R_world     = f["transforms"]["rightHand"][t]

    L_cam = _pose_world_to_cam(L_world, T_world_cam)
    R_cam = _pose_world_to_cam(R_world, T_world_cam)

    L_pos = L_cam[:3, 3].astype(np.float32)
    R_pos = R_cam[:3, 3].astype(np.float32)
    L_r6  = _rotmat_to_rot6d(L_cam[:3, :3])
    R_r6  = _rotmat_to_rot6d(R_cam[:3, :3])

    hands_pad = np.zeros((12,), dtype=np.float32)
    return np.concatenate([L_pos, L_r6, [0.0], R_pos, R_r6, [0.0], hands_pad], dtype=np.float32)

def read_rgb_resized(mp4_path: str, t: int, size: int = 224) -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {t} from {mp4_path}")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb  # uint8 [0,255]

# ───────── core eval ─────────
def sample_random_episode(root: str, part: str, task: str) -> tuple[str, str, int]:
    """
    Returns (h5_path, mp4_path, N) for a random episode under root/part/task.
    Ensures the pair exists and N > 0.
    """
    task_dir = os.path.join(root, part, task)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task folder not found: {task_dir}")

    h5_files = sorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
    pairs = []
    for h5f in h5_files:
        mp4f = h5f.replace(".hdf5", ".mp4")
        if os.path.exists(mp4f):
            try:
                with h5py.File(h5f, "r") as f:
                    N = int(f["transforms"]["leftHand"].shape[0])
                if N > 0:
                    pairs.append((h5f, mp4f, N))
            except Exception:
                pass
    if not pairs:
        raise RuntimeError(f"No valid (hdf5, mp4) pairs in {task_dir}")
    return random.choice(pairs)

def run_eval(root, part, task, ckpt_dir, img_size=224, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Choose episode + random start
    h5_path, mp4_path, N = sample_random_episode(root, part, task)

    # Load config & policy
    conf = cfg.get_config("pi0_egodex")
    H    = int(conf.model.action_horizon)
    ckpt_dir = download.maybe_download(ckpt_dir)
    policy   = policy_config.create_trained_policy(conf, ckpt_dir)

    # Pick a random start t0 that can fit up to H frames; we’ll clamp later just in case
    t0_max = max(0, N - 1)
    # prefer to have window if possible
    t0 = random.randint(0, max(0, N - H)) if N >= H else random.randint(0, t0_max)

    # Build example at t0
    img_u8     = read_rgb_resized(mp4_path, t0, size=img_size)
    state_vec  = None
    with h5py.File(h5_path, "r") as f:
        state_vec = make_pi0_state_vec(f, t0)

    prompt = os.path.basename(os.path.dirname(h5_path))  # task folder name

    example = {
        "state":  jnp.asarray(state_vec),            # (32,)
        "image":  jnp.asarray(img_u8.astype(np.uint8)),
        "prompt": prompt,
    }

    # Inference
    pred = np.asarray(policy.infer(example)["actions"])  # (T_pred, 32)
    T_pred = pred.shape[0]
    # Effective length available in GT from t0
    T_avail = N - t0
    T = min(T_pred, T_avail)
    pred = pred[:T]

    # Build GT wrist trajectories (camera frame) for T steps
    traj_gt_L = np.zeros((T, 3), dtype=np.float32)
    traj_gt_R = np.zeros((T, 3), dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        for i in range(T):
            t = t0 + i
            T_world_cam = f["transforms"]["camera"][t]
            L_world     = f["transforms"]["leftHand"][t]
            R_world     = f["transforms"]["rightHand"][t]
            L_cam = _pose_world_to_cam(L_world, T_world_cam)
            R_cam = _pose_world_to_cam(R_world, T_world_cam)
            traj_gt_L[i] = L_cam[:3, 3]
            traj_gt_R[i] = R_cam[:3, 3]

    # Predicted wrist xyz slices under 32-D layout
    traj_pred_L = pred[:, 0:3]       # left pos
    traj_pred_R = pred[:, 10:13]     # right pos (after 3+6+1=10)

    # ───────── plot ─────────
    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(*traj_pred_L.T, label="pred left wrist")
    ax.plot(*traj_gt_L.T,   label="GT left wrist",   linestyle="--")

    ax.plot(*traj_pred_R.T, label="pred right wrist")
    ax.plot(*traj_gt_R.T,   label="GT right wrist",  linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"EgoDex — {part}/{task} — episode @ t0={t0}\nPred vs GT wrist trajectories")
    ax.legend()
    ax.set_box_aspect([1,1,1])

    out_png = f"egodex_wrist_traj_{part}_{task}_t0_{t0}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Episode file:", h5_path)
    print("Video file:  ", mp4_path)
    print("N frames:    ", N)
    print("t0 chosen:   ", t0)
    print("T_pred:      ", T_pred)
    print("T_used:      ", T)
    print("Saved plot:  ", out_png)

# ───────── CLI ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EgoDex root (contains part1/part2/...)")
    ap.add_argument("--part", required=True, help="Folder name under root, e.g., part1/part2/test/extra")
    ap.add_argument("--task", required=True, help="Task folder name under the part directory")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path or model hub spec")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    run_eval(
        root=args.root,
        part=args.part,
        task=args.task,
        ckpt_dir=args.ckpt,
        img_size=args.img_size,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
