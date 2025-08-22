"""
USAGE: 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_policy.py

Galaxea-pi0 inference + 3-D comparison plot
──────────────────────────────────────────
• Uses the *real* proprio state from the same CSV row as the image.
• Plots wrist-position trajectories (left: dims 0-2, right: dims 7-9)
  for both prediction and ground truth, then saves to disk.
• Adjust paths, image resize size, and _STATE_COLS list if you changed them.
"""

# ─────────────────────────── imports ────────────────────────────
import cv2
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.shared import download


# ─────────────—— file locations & basic constants —──────────────
idx = 0
IMG_PATH = f"/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/left/000000.jpg"
CSV_PATH = f"/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/ee_pos/ee_poses_and_hands.csv"
PROMPT   = "pick up can and move it to the center"
IMG_SIZE = 224              # use 224 if that’s what you trained on

# All 26 state/action columns in order
_STATE_COLS = [
    "left_pos_x", "left_pos_y", "left_pos_z",
    "left_ori_x", "left_ori_y", "left_ori_z", "left_ori_w",
    "right_pos_x", "right_pos_y", "right_pos_z",
    "right_ori_x", "right_ori_y", "right_ori_z", "right_ori_w",
    "left_hand_0", "left_hand_1", "left_hand_2", "left_hand_3",
    "left_hand_4", "left_hand_5",
    "right_hand_0", "right_hand_1", "right_hand_2", "right_hand_3",
    "right_hand_4", "right_hand_5",
]

# ─────────────────────── load image ─────────────────────────────
img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Cannot read {IMG_PATH}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

# ────────────────────── load CSV & state ────────────────────────
df   = pd.read_csv(CSV_PATH)
row0 = df.iloc[0]               # take first frame; change if needed

# Build 26-D state vector in the *same order* used during training
state_vec = row0[_STATE_COLS].to_numpy(dtype=np.float32)

# ─────────────── build inference example dict ───────────────────
example = {
    "state": state_vec,
    "image": img.astype(np.uint8),
    "prompt": PROMPT,
}
example = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in example.items()}

# ──────────────────── load policy & infer ───────────────────────
conf      = cfg.get_config("pi0_galaxea")
ckpt_dir  = download.maybe_download("checkpoints/pi0_galaxea/my_experiment/8000")
policy    = policy_config.create_trained_policy(conf, ckpt_dir)

pred = np.asarray(policy.infer(example)["actions"])   # shape (T, 26)

# ───────────────────── get matching ground truth ─────────────────
# The CSV is 30 fps; model actions are per 2 frames in your earlier code,
# so we sub-sample every 2nd row to align lengths.
gt_rows = df.iloc[idx : idx + len(pred)]

gt = gt_rows[_STATE_COLS].to_numpy(dtype=np.float32)      # shape (T, 26)

# If the model outputs *delta* actions but CSV is absolute, integrate:
# pred = np.cumsum(pred, axis=0)

# ───────────────────── slice wrist trajectories ─────────────────
# left wrist xyz  = dims 0-2, right wrist xyz = dims 7-9
traj_pred_L = pred[:, 0:3]
traj_pred_R = pred[:, 7:10]

# reference (absolute) wrist positions from the starting frame
# ref_row = df.iloc[idx]
# ref_L = ref_row[["left_pos_x","left_pos_y","left_pos_z"]].to_numpy(dtype=np.float32)   # (3,)
# ref_R = ref_row[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy(dtype=np.float32) # (3,)

# # ───────────────────── slice + convert deltas → absolute ─────────
# # predicted deltas
# dL = pred[:, 0:3]      # left wrist deltas
# dR = pred[:, 7:10]     # right wrist deltas

# # add reference to get absolute trajectories
# traj_pred_L = dL + ref_L[None, :]
# traj_pred_R = dR + ref_R[None, :]

traj_gt_L   = gt[:, 0:3]
traj_gt_R   = gt[:, 7:10]

# ─────────────────────────── plot & save ────────────────────────
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")

ax.plot(*traj_pred_L.T, label="pred left wrist",  color="tab:blue")
ax.plot(*traj_gt_L.T,   label="GT left wrist",    color="tab:blue",  linestyle="--")

ax.plot(*traj_pred_R.T, label="pred right wrist", color="tab:orange")
ax.plot(*traj_gt_R.T,   label="GT right wrist",   color="tab:orange", linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Predicted vs Ground-Truth Wrist Trajectories")
ax.legend()
ax.set_box_aspect([1, 1, 1])       # equal axis scaling

fig.savefig("wrist_traj_pred_vs_gt.png", dpi=300, bbox_inches="tight")
plt.close(fig)
