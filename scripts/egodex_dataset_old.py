import os, glob, cv2, h5py, math
import numpy as np
from bisect import bisect_right
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
    # [r00, r10, r20, r01, r11, r21] (first two columns, column-major stacking)
    return np.array([Rm[0,0], Rm[1,0], Rm[2,0], Rm[0,1], Rm[1,1], Rm[2,1]], dtype=np.float32)

def _pose_world_to_cam(T_world_obj: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    # Convert a world pose to the camera frame at the same timestep.
    T_cam_world = _inv(T_world_cam)
    return T_cam_world @ T_world_obj

# ---------- dataset ----------
class EgoDexLeRobotLikeDataset(Dataset):
    """
    Map-style dataset that returns a single frame:
      sample = {
        "image":  (H, W, 3) uint8 RGB resized to image_size,
        "state":  (D,) float32,
        "actions":(D,) float32,
        "task":   str (from HDF5 attrs['llm_description'] if present)
      }

    state_format:
      - "pi0": [left_xyz(3), left_rot6d(6), 0.0, right_xyz(3), right_rot6d(6), 0.0, zeros(12)] -> 32D
      - "ego": [L_wrist xyz+quat(7), R_wrist xyz+quat(7), 10 fingertips xyz (30)] -> 44D
    """
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        state_format: Literal["pi0", "ego"] = "pi0",
        sample_every_k: int = 1,         # optionally subsample frames: use every k-th frame
        traj_per_task: Optional[int] = None,   # optional cap per task
        max_episodes: Optional[int] = None,    # optional global cap for quick smoke tests
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.state_format = state_format
        self.sample_every_k = max(1, int(sample_every_k))

        # joints we might use
        self.wrists = ["leftHand", "rightHand"]
        self.fingertips = [
            "leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip", "leftRingFingerTip", "leftLittleFingerTip",
            "rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip",
        ]

        # 1) collect (h5, mp4) pairs by part/task
        self.episodes: List[Tuple[str, str, int]] = []  # (h5_path, mp4_path, length_N)
        part_dirs = sorted(
            d for d in os.listdir(root_dir)
            if d.startswith("part") and os.path.isdir(os.path.join(root_dir, d))
        )
        # include 'test' or 'extra' if present
        for extra in ("test", "extra"):
            p = os.path.join(root_dir, extra)
            if os.path.isdir(p):
                part_dirs.append(extra)

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
                # optional per-task cap
                if traj_per_task is not None and len(pairs) > traj_per_task:
                    idxs = np.random.choice(len(pairs), size=traj_per_task, replace=False)
                    pairs = [pairs[i] for i in idxs]
                for h5f, mp4f in pairs:
                    try:
                        with h5py.File(h5f, "r") as f:
                            N = int(f["transforms"]["leftHand"].shape[0])
                        if N > 0:
                            self.episodes.append((h5f, mp4f, N))
                    except Exception:
                        # skip unreadable files
                        continue

        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]

        if not self.episodes:
            raise RuntimeError(f"No valid (hdf5, mp4) pairs found under {root_dir}")

        # 2) build cumulative frame offsets (using subsampling factor)
        self.lengths = [math.floor(N / self.sample_every_k) for (_, _, N) in self.episodes]
        self.cum = np.cumsum([0] + self.lengths)  # shape (M+1,)
        self.total = int(self.cum[-1])


    def __len__(self) -> int:
        return self.total

    def _map_index(self, idx: int) -> Tuple[int, int]:
        # global idx -> (episode_id, local_frame_t)
        # local frame = (idx_offset * sample_every_k)
        ep_id = bisect_right(self.cum, idx) - 1
        offset = idx - self.cum[ep_id]
        t = offset * self.sample_every_k
        return ep_id, t

    def _load_frame_image(self, mp4_path: str, t: int) -> np.ndarray:
        cap = cv2.VideoCapture(mp4_path)
        # Random access: set the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {t} from {mp4_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, self.image_size, interpolation=cv2.INTER_AREA)
        return frame_rgb  # uint8, HxWx3

    def _state_pi0(self, f: h5py.File, t: int) -> np.ndarray:
        # wrist poses in camera frame, with Rot6D; pad hands with zeros(12)
        T_world_cam = f["transforms"]["camera"][t]  # camera in world frame
        L_world = f["transforms"]["leftHand"][t]
        R_world = f["transforms"]["rightHand"][t]

        L_cam = _pose_world_to_cam(L_world, T_world_cam)
        R_cam = _pose_world_to_cam(R_world, T_world_cam)

        L_pos = L_cam[:3, 3].astype(np.float32)
        R_pos = R_cam[:3, 3].astype(np.float32)
        L_rot6 = _rotmat_to_rot6d(L_cam[:3, :3])
        R_rot6 = _rotmat_to_rot6d(R_cam[:3, :3])

        hands_pad = np.zeros((12,), dtype=np.float32)  # dataset has no robot hand joints
        state = np.concatenate([L_pos, L_rot6, [0.0], R_pos, R_rot6, [0.0], hands_pad]).astype(np.float32)
        return state

    def _state_ego(self, f: h5py.File, t: int) -> np.ndarray:
        # wrists xyz+quat (x,y,z,w) in camera frame + 10 fingertips xyz
        T_world_cam = f["transforms"]["camera"][t]
        # wrists
        ws = []
        for joint in self.wrists:
            T_world = f["transforms"][joint][t]
            T_cam   = _pose_world_to_cam(T_world, T_world_cam)
            pos = T_cam[:3, 3].astype(np.float32)
            quat = R.from_matrix(T_cam[:3, :3]).as_quat().astype(np.float32)  # (x,y,z,w)
            ws.append(np.concatenate([pos, quat], dtype=np.float32))
        wrists_vec = np.concatenate(ws, dtype=np.float32)  # 14

        # fingertips (xyz only)
        tips = []
        for joint in self.fingertips:
            T_world = f["transforms"][joint][t]
            T_cam   = _pose_world_to_cam(T_world, T_world_cam)
            tips.append(T_cam[:3, 3].astype(np.float32))
        tips_vec = np.concatenate(tips, dtype=np.float32)  # 30

        return np.concatenate([wrists_vec, tips_vec], dtype=np.float32)  # 44

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, t = self._map_index(idx)
        h5_path, mp4_path, N = self.episodes[ep_id]

        # guard in case rounding pushes t==N
        if t >= N:
            t = N - 1

        # image
        image = self._load_frame_image(mp4_path, t)  # uint8 HxWx3

        # state/actions
        with h5py.File(h5_path, "r") as f:
            if self.state_format == "ego":
                state = self._state_ego(f, t)
            else:
                state = self._state_pi0(f, t)

            task = ""
            try:
                task = f.attrs.get("llm_description", "")
                # some samples may have a second reversible description
                if isinstance(task, bytes):
                    task = task.decode("utf-8", errors="ignore")
            except Exception:
                task = ""

        sample = {
            "image": image,                    # np.uint8, (H,W,3)
            "state": state.astype(np.float32), # (D,)
            "actions": state.astype(np.float32),
            "task": task,
        }
        return sample


def create_torch_dataset(data_config, action_horizon, model_config):
    if getattr(data_config, "repo_id", None) == "egodex":
        # Expect data_config.egodex_root, .state_format, .sample_every_k, etc.
        
        egodex_root = "/iris/projects/humanoid/dataset/ego_dex"
        return EgoDexLeRobotLikeDataset(
            # root_dir=str(data_config.egodex_root),
            root_dir=str(egodex_root),
            image_size=(224, 224),
            state_format=getattr(data_config, "state_format", "pi0"),
            sample_every_k=getattr(data_config, "sample_every_k", 1),
            traj_per_task=getattr(data_config, "traj_per_task", None),
            max_episodes=getattr(data_config, "max_episodes", None),
        )
    # else fall back to LeRobot:
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={k: [t / dataset_meta.fps for t in range(action_horizon)]
                          for k in data_config.action_sequence_keys},
    )
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset

import os
import argparse
import dataclasses
import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

def build_data_config(train_cfg, repo_id: str, prompt_from_task: bool = True):
    # Start from the config's data factory, then create a concrete DataConfig
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    # Override only what we need for this test
    # norm_stats is not needed to test create_torch_dataset (only used during transform)
    data_cfg = dataclasses.replace(
        data_cfg,
        repo_id=repo_id,
        prompt_from_task=prompt_from_task,
    )
    return data_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="fake",
                        help="HF LeRobot repo_id or 'fake' for a synthetic dataset")
    parser.add_argument("--hf_home", type=str, default=None,
                        help="Path for HF_LEROBOT_HOME (root that contains your repo folder)")
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_LEROBOT_HOME"] = args.hf_home

    # Load a training preset just to get model + assets + action_horizon
    train_cfg = _config.get_config("pi0_galaxea")
    data_cfg = build_data_config(train_cfg, repo_id=args.repo)

    # This is the function under test
    dataset = _data_loader.create_torch_dataset(
        data_cfg,
        action_horizon=train_cfg.model.action_horizon,
        model_config=train_cfg.model,
    )

    print(f"type(dataset) = {type(dataset)}")
    try:
        n = len(dataset)
        print(f"len(dataset) = {n}")
    except Exception as e:
        print("len(dataset) raised:", repr(e))

    # Probe a few indices that usually exist
    # probe_idxs = [0, 1, 10]
    # for i in probe_idxs:
    #     try:
    #         sample = dataset[i]
    #         print(f"\nSample {i}: keys = {list(sample.keys())}")
    #         if "image" in sample:
    #             img = sample["image"]
    #             print("  image:", getattr(img, "shape", type(img)), getattr(img, "dtype", type(img)))
    #         if "state" in sample:
    #             st = sample["state"]
    #             print("  state:", getattr(st, "shape", type(st)), getattr(st, "dtype", type(st)))
    #             if hasattr(st, "shape") and st.size:
    #                 st_np = np.asarray(st)
    #                 print("    state min/max:", float(st_np.min()), float(st_np.max()))
    #         if "actions" in sample:
    #             ac = sample["actions"]
    #             print("  actions:", getattr(ac, "shape", type(ac)), getattr(ac, "dtype", type(ac)))
    #         if "task" in sample:
    #             print("  task:", sample["task"])
    #     except Exception as e:
    #         print(f"  fetching sample {i} raised:", repr(e))

if __name__ == "__main__":
    main()