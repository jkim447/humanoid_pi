"""
USAGE:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_policy_egodex_multi.py \
  --root /iris/projects/humanoid/dataset/ego_dex \
  --ckpt checkpoints/pi0_galaxea/my_egodex_experiment/10000 \
  --parts part1 part2 part3 \
  --tasks_per_part 1 \
  --waypoints 50 \
  --out_dir egodex_eval_out \
  --img_size 224 \
  --seed 123
"""

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

# ---------- camera intrinsics (given for 1920x1080) ----------
K_NATIVE = np.array([
    [736.6339,   0.0, 960.0],
    [  0.0,    736.6339, 540.0],
    [  0.0,      0.0,     1.0 ],
], dtype=np.float32)

# ---------- SE(3) helpers ----------
def _inv(T: np.ndarray) -> np.ndarray:
    Rm, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = Rm.T
    inv[:3, 3]  = -Rm.T @ t
    return inv

def _pose_world_to_cam(T_world_obj: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    return _inv(T_world_cam) @ T_world_obj

def _rotmat_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    return np.array([Rm[0,0], Rm[1,0], Rm[2,0], Rm[0,1], Rm[1,1], Rm[2,1]], dtype=np.float32)

def make_pi0_state_vec(f: h5py.File, t: int) -> np.ndarray:
    """
    32-D layout: [L_xyz(3), L_rot6d(6), 0.0, R_xyz(3), R_rot6d(6), 0.0, zeros(12)]
    """
    T_world_cam = f["transforms"]["camera"][t]
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

# ---------- IO helpers ----------
def read_rgb_resized(mp4_path: str, t: int, size: int = 224) -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {t} from {mp4_path}")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb  # uint8

def list_tasks(root: str, part: str):
    part_dir = os.path.join(root, part)
    if not os.path.isdir(part_dir):
        return []
    return sorted([d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))])

def list_all_tasks(root: str, parts: list[str]) -> list[str]:
    """Union of task names across all selected parts."""
    tasks = set()
    for part in parts:
        part_dir = os.path.join(root, part)
        if not os.path.isdir(part_dir):
            continue
        for d in os.listdir(part_dir):
            if os.path.isdir(os.path.join(part_dir, d)):
                tasks.add(d)
    return sorted(tasks)

def sample_random_episode(root: str, part: str, task: str):
    task_dir = os.path.join(root, part, task)
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

# ---------- projection ----------
def scale_intrinsics(K_native: np.ndarray, out_w: int, out_h: int, native_w: int = 1920, native_h: int = 1080):
    sx = out_w / float(native_w)
    sy = out_h / float(native_h)
    K = K_native.copy()
    K[0,0] *= sx  # fx
    K[1,1] *= sy  # fy
    K[0,2] *= sx  # cx
    K[1,2] *= sy  # cy
    return K

def project_points_cam_to_img(X_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    X_cam: (T,3) camera-frame points
    Returns (T,2) pixel coords
    """
    X = X_cam.astype(np.float32)
    zs = np.clip(X[:, 2:3], 1e-6, None)  # avoid div-by-zero
    uvw = X @ K[:3, :3].T
    uv  = uvw[:, :2] / zs
    return uv

# ---------- plotting ----------
def plot_3d(traj_pred_L, traj_gt_L, traj_pred_R, traj_gt_R, title, out_path):
    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(*traj_pred_L.T, label="pred left wrist")
    ax.plot(*traj_gt_L.T,   label="GT left wrist",   linestyle="--")

    ax.plot(*traj_pred_R.T, label="pred right wrist")
    ax.plot(*traj_gt_R.T,   label="GT right wrist",  linestyle="--")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_box_aspect([1,1,1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_on_image(img_rgb_u8, uv_pred_L, uv_gt_L, uv_pred_R, uv_gt_R, title, out_path):
    """
    Draw trajectories on the resized RGB image.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_rgb_u8)
    # Left wrist
    ax.plot(uv_pred_L[:,0], uv_pred_L[:,1], label="pred L", linewidth=2)
    ax.plot(uv_gt_L[:,0],   uv_gt_L[:,1],   "--", label="GT L", linewidth=2)
    # Right wrist
    ax.plot(uv_pred_R[:,0], uv_pred_R[:,1], label="pred R", linewidth=2)
    ax.plot(uv_gt_R[:,0],   uv_gt_R[:,1],   "--", label="GT R", linewidth=2)

    # draw points to emphasize waypoints
    ax.scatter(uv_pred_L[:,0], uv_pred_L[:,1], s=12)
    ax.scatter(uv_gt_L[:,0],   uv_gt_L[:,1],   s=12)
    ax.scatter(uv_pred_R[:,0], uv_pred_R[:,1], s=12)
    ax.scatter(uv_gt_R[:,0],   uv_gt_R[:,1],   s=12)

    ax.set_title(title)
    ax.set_xlim([0, img_rgb_u8.shape[1]])
    ax.set_ylim([img_rgb_u8.shape[0], 0])  # image coords: origin top-left
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- per-task evaluation ----------
def evaluate_one_task(conf, policy, root, part, task, waypoints, img_size, out_dir):
    # episode + random start
    h5_path, mp4_path, N = sample_random_episode(root, part, task)
    H = int(conf.model.action_horizon)

    # choose t0; aim to have at least H frames, but clamp later
    t0 = random.randint(0, max(0, N - H)) if N >= H else random.randint(0, N-1)

    # inputs at t0
    img_u8 = read_rgb_resized(mp4_path, t0, size=img_size)
    with h5py.File(h5_path, "r") as f:
        state_vec = make_pi0_state_vec(f, t0)

    prompt = task  # use folder name as prompt
    example = {
        "state":  jnp.asarray(state_vec),           # (32,)
        "image":  jnp.asarray(img_u8.astype(np.uint8)),
        "prompt": prompt,
    }

    # inference
    pred = np.asarray(policy.infer(example)["actions"])  # (T_pred, 32)
    T_pred   = pred.shape[0]
    T_avail  = N - t0
    T        = min(waypoints, T_pred, T_avail)          # <-- limit to 50 (or user-specified)
    pred     = pred[:T]

    # build GT for T steps (camera frame)
    gt_L = np.zeros((T,3), dtype=np.float32)
    gt_R = np.zeros((T,3), dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        for i in range(T):
            t = t0 + i
            T_world_cam = f["transforms"]["camera"][t]
            L_world     = f["transforms"]["leftHand"][t]
            R_world     = f["transforms"]["rightHand"][t]
            L_cam = _pose_world_to_cam(L_world, T_world_cam)
            R_cam = _pose_world_to_cam(R_world, T_world_cam)
            gt_L[i] = L_cam[:3, 3]
            gt_R[i] = R_cam[:3, 3]

    # predicted wrist xyz (32-D layout)
    pred_L = pred[:, 0:3]
    pred_R = pred[:, 10:13]  # after 3 + 6 + 1

    # 3D plot
    title3d = f"{part}/{task} — episode @ t0={t0} — T={T}"
    out3d   = os.path.join(out_dir, f"traj3d_{part}_{task}_t0_{t0}_T{T}.png")
    plot_3d(pred_L, gt_L, pred_R, gt_R, title3d, out3d)

    # 2D image-plane plot (project to resized image)
    K = scale_intrinsics(K_NATIVE, out_w=img_size, out_h=img_size)  # scaled to (img_size, img_size)
    uv_pred_L = project_points_cam_to_img(pred_L, K)
    uv_pred_R = project_points_cam_to_img(pred_R, K)
    uv_gt_L   = project_points_cam_to_img(gt_L,   K)
    uv_gt_R   = project_points_cam_to_img(gt_R,   K)

    title2d = f"{part}/{task} — image-plane — t0={t0} — T={T}"
    out2d   = os.path.join(out_dir, f"traj2d_{part}_{task}_t0_{t0}_T{T}.png")
    plot_on_image(img_u8, uv_pred_L, uv_gt_L, uv_pred_R, uv_gt_R, title2d, out2d)

    print(f"[{part}/{task}] N={N}, t0={t0}, T_pred={T_pred}, T_used={T} -> saved:")
    print("  3D :", out3d)
    print("  2D :", out2d)

# ---------- main loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EgoDex root (contains part1/part2/...)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path or model hub spec")
    ap.add_argument("--parts", nargs="+", default=["part1","part2","part3"], help="Parts to sample from")
    ap.add_argument("--tasks_per_part", type=int, default=1, help="How many random tasks per part to evaluate")
    ap.add_argument("--waypoints", type=int, default=50, help="Number of GT/pred waypoints to plot")
    ap.add_argument("--out_dir", type=str, default="egodex_eval_out", help="Folder to save plots")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # config & policy
    conf = cfg.get_config("pi0_egodex")   # <- your egodex-trained model config
    ckpt_dir = download.maybe_download(args.ckpt)
    policy   = policy_config.create_trained_policy(conf, ckpt_dir)

    # loop parts -> random tasks
    for part in args.parts:
        # loop tasks (one trajectory per unique task across all parts)
        all_tasks = list_all_tasks(args.root, args.parts)
        if not all_tasks:
            print("[WARN] no tasks found in the selected parts")
            return

        # If you still want to limit how many tasks you evaluate, reuse --tasks_per_part
        # as a total cap (optional). Remove these two lines if you want *all* tasks.
        pick = min(args.tasks_per_part, len(all_tasks))
        all_tasks = random.sample(all_tasks, pick)

        for task in all_tasks:
            # choose any part that has this task
            candidate_parts = [p for p in args.parts if os.path.isdir(os.path.join(args.root, p, task))]
            if not candidate_parts:
                print(f"[WARN] task '{task}' not found under any of {args.parts}; skipping")
                continue
            part = random.choice(candidate_parts)

            try:
                evaluate_one_task(
                    conf=conf,
                    policy=policy,
                    root=args.root,
                    part=part,            # used only for titles/filenames
                    task=task,            # one trajectory per task
                    waypoints=args.waypoints,
                    img_size=args.img_size,
                    out_dir=args.out_dir,
                )
            except Exception as e:
                print(f"[ERROR] {task}: {e}")


if __name__ == "__main__":
    main()
