"""
Usage (for testing):
uv run src/openpi/training/egodex_dataset.py
"""
import os, glob, cv2, h5py
import numpy as np
from typing import List, Tuple, Dict, Any, Literal, Optional
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

# ---------- small SE(3) helpers ----------
def _inv(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t  = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = Rm.T
    inv[:3, 3]  = -Rm.T @ t
    return inv

def _rotmat_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    # [r00, r10, r20, r01, r11, r21]
    return np.array([Rm[0,0], Rm[1,0], Rm[2,0], Rm[0,1], Rm[1,1], Rm[2,1]], dtype=np.float32)

def _pose_world_to_cam(T_world_obj: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    return _inv(T_world_cam) @ T_world_obj

# ---------- simplified, windowed dataset ----------
class EgoDexSeqDataset(Dataset):
    """
    Map-style dataset that yields *windows*:
      sample = {
        "image":   (H, W, 3) uint8 RGB at start frame (resized to image_size),
        "state":   (D,) float32  — state at start frame,
        "actions": (H, D) float32 — sequence of H = action_horizon states,
        "task":    str (from HDF5 attrs['llm_description'] if present)
      }

    state_format:
      - "pi0": [L_xyz(3), L_rot6d(6), 0.0, R_xyz(3), R_rot6d(6), 0.0, zeros(12)] -> 32-D
      - "ego": [L (xyz+quat 7), R (xyz+quat 7), 10 fingertips xyz (30)] -> 44-D
    """
    def __init__(
        self,
        root_dir: str,
        action_horizon: int,
        image_size: tuple[int, int] = (224, 224),
        state_format: Literal["pi0", "ego"] = "ego", #"pi0",
        window_stride: int = 1,                 # step for consecutive windows
        traj_per_task: Optional[int] = None,    # optional cap per task
        max_episodes: Optional[int] = None,     # optional global cap
    ):
        assert action_horizon >= 1
        self.root_dir = root_dir
        self.image_size = image_size
        self.state_format = state_format
        self.H = int(action_horizon)
        self.stride = max(1, int(window_stride))

        self.wrists = ["leftHand", "rightHand"]
        self.fingertips = [
            "leftThumbTip","leftIndexFingerTip","leftMiddleFingerTip","leftRingFingerTip","leftLittleFingerTip",
            "rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip",
        ]

        # 1) collect (h5, mp4, N) per episode
        episodes: List[Tuple[str, str, int]] = []
        part_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith("part") or d in ("test","extra"))
        )
        for part in part_dirs:
            part_path = os.path.join(root_dir, part)
            for task in sorted(os.listdir(part_path)):
                task_path = os.path.join(part_path, task)
                if not os.path.isdir(task_path): 
                    continue
                h5_files = sorted(glob.glob(os.path.join(task_path, "*.hdf5")))
                pairs = []
                for h5f in h5_files:
                    mp4f = h5f.replace(".hdf5", ".mp4")
                    if os.path.exists(mp4f):
                        pairs.append((h5f, mp4f))
                if traj_per_task is not None and len(pairs) > traj_per_task:
                    idxs = np.random.choice(len(pairs), size=traj_per_task, replace=False)
                    pairs = [pairs[i] for i in idxs]
                for h5f, mp4f in pairs:
                    try:
                        with h5py.File(h5f, "r") as f:
                            N = int(f["transforms"]["leftHand"].shape[0])
                        if N >= self.H:
                            episodes.append((h5f, mp4f, N))
                    except Exception:
                        continue

        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        if not episodes:
            raise RuntimeError(f"No valid (hdf5, mp4) pairs with at least {self.H} frames under {root_dir}")

        self.episodes = episodes

        # 2) precompute flat list of valid window starts: (ep_id, t0)
        #    windows start at t0 = 0, stride, 2*stride, ..., <= N - H
        index = []
        for ep_id, (_, _, N) in enumerate(self.episodes):
            last_start = N - self.H
            index.extend((ep_id, t0) for t0 in range(0, last_start + 1, self.stride))
        self.index: List[Tuple[int,int]] = index

    def __len__(self) -> int:
        return len(self.index)

    # ---- state builders ----
    def _state_pi0(self, f: h5py.File, t: int) -> np.ndarray:
        T_world_cam = f["transforms"]["camera"][t]
        L_world = f["transforms"]["leftHand"][t]
        R_world = f["transforms"]["rightHand"][t]
        L_cam = _pose_world_to_cam(L_world, T_world_cam)
        R_cam = _pose_world_to_cam(R_world, T_world_cam)
        L_pos = L_cam[:3, 3].astype(np.float32)
        R_pos = R_cam[:3, 3].astype(np.float32)
        L_rot6 = _rotmat_to_rot6d(L_cam[:3, :3])
        R_rot6 = _rotmat_to_rot6d(R_cam[:3, :3])
        hands_pad = np.zeros((12,), dtype=np.float32)
        return np.concatenate([L_pos, L_rot6, [0.0], R_pos, R_rot6, [0.0], hands_pad], dtype=np.float32)

    def _state_ego(self, f: h5py.File, t: int) -> np.ndarray:
        T_world_cam = f["transforms"]["camera"][t]
        # wrists xyz+quat
        ws = []
        for joint in self.wrists:
            T_world = f["transforms"][joint][t]
            T_cam   = _pose_world_to_cam(T_world, T_world_cam)
            pos  = T_cam[:3, 3].astype(np.float32)
            rot6 = _rotmat_to_rot6d(T_cam[:3, :3])             # (6,) from first two R columns
            ws.append(np.concatenate([pos, rot6], dtype=np.float32))
        wrists_vec = np.concatenate(ws, dtype=np.float32)  # 18
        # fingertips xyz
        tips = []
        for joint in self.fingertips:
            T_world = f["transforms"][joint][t]
            T_cam   = _pose_world_to_cam(T_world, T_world_cam)
            tips.append(T_cam[:3, 3].astype(np.float32))
        tips_vec = np.concatenate(tips, dtype=np.float32)  # 30
        return np.concatenate([wrists_vec, tips_vec], dtype=np.float32)  # 48

    def _state_vec(self, f: h5py.File, t: int) -> np.ndarray:
        return self._state_ego(f, t) if self.state_format == "ego" else self._state_pi0(f, t)

    # ---- IO helpers ----
    def _read_rgb(self, mp4_path: str, t: int) -> np.ndarray:
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {t} from {mp4_path}")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0  # normalize to 0–1
        return rgb  # float32

    # ---- main fetch ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, t0 = self.index[idx]
        h5_path, mp4_path, N = self.episodes[ep_id]

        # image at start frame (you can switch to center frame if you prefer)
        image = self._read_rgb(mp4_path, t0)

        with h5py.File(h5_path, "r") as f:
            # sequence of states for actions
            actions = np.stack([self._state_vec(f, t0 + dt) for dt in range(self.H)], axis=0)  # (H, D)
            state   = actions[0].copy()  # state at start

        # task = folder name (parent directory of the file)
        task = os.path.basename(os.path.dirname(h5_path))

        return {
            "image":   image,          # uint8, (H,W,3)
            "state":   state.astype(np.float32),      # (D,)
            "actions": actions.astype(np.float32),    # (H,D)
            "task":    task,
        }
############################################
############################################
# RUN BELOW TO TEST THE DATASET
############################################
############################################

# import os
# import random
# import cv2
# from pathlib import Path
# import numpy as np
# import torch

# # === configuration ===
# root_dir = "/iris/projects/humanoid/dataset/ego_dex"   # <-- change to your dataset root
# save_dir = Path("dataset_test_images")
# save_dir.mkdir(exist_ok=True)

# # === create dataset ===
# ds = EgoDexSeqDataset(
#     root_dir=root_dir,
#     action_horizon=5,        # example horizon
#     image_size=(224, 224),
#     state_format="ego",      # or "ego"
#     window_stride=1,
#     traj_per_task = 1
# )

# print(f"Dataset length: {len(ds)} samples")

# # === test retrieval ===
# num_samples_to_test = 5
# indices = random.sample(range(len(ds)), num_samples_to_test)

# for idx in indices:
#     sample = ds[idx]
#     image = sample["image"]
#     state = sample["state"]
#     actions = sample["actions"]
#     task = sample["task"]

#     print(f"Sample {idx}:")
#     print(f"  Image shape:   {image.shape}  dtype={image.dtype}")
#     print(f"  State shape:   {state.shape}")
#     print(f"  Actions shape: {actions.shape}")
#     print(f"  Task:          {task}")

#     # save image as RGB
#     img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     img_bgr_u8 = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
#     save_path = save_dir / f"sample_{idx}_{task}.png"
#     cv2.imwrite(str(save_path), img_bgr_u8)
#     print(f"  Saved image to {save_path}")

# print(f"\nDone! Check images in {save_dir.resolve()}")