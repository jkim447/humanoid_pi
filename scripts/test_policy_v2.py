"""
USAGE:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_policy_v2.py
"""

# ───────── imports ─────────
import cv2, pandas as pd, numpy as np, jax.numpy as jnp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.shared import download

# ───────── paths & constants ─────────
idx       = 0
IMG_PATH  = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/left/000000.jpg"
CSV_PATH  = "/iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20/Demo1/ee_pos/ee_poses_and_hands.csv"
PROMPT    = "pick up can and move it to the center" # I changed this pre-emptively (removed the control mode tags)
IMG_SIZE  = 224
ACTION_DIM = 32  # (3+6+1)*2 + 12
model_path = "checkpoints/pi0_galaxea/my_experiment_08122025/10000"

# new 32-D state names
_STATE_COLS_32 = [
    # left
    "left_pos_x","left_pos_y","left_pos_z",
    "left_rot6d_0","left_rot6d_1","left_rot6d_2","left_rot6d_3","left_rot6d_4","left_rot6d_5",
    "left_pad",
    # right
    "right_pos_x","right_pos_y","right_pos_z",
    "right_rot6d_0","right_rot6d_1","right_rot6d_2","right_rot6d_3","right_rot6d_4","right_rot6d_5",
    "right_pad",
    # hands
    "left_hand_0","left_hand_1","left_hand_2","left_hand_3","left_hand_4","left_hand_5",
    "right_hand_0","right_hand_1","right_hand_2","right_hand_3","right_hand_4","right_hand_5",
]

def quat_xyzw_to_rot6d(q_xyzw: np.ndarray) -> np.ndarray:
    Rm = R.from_quat(q_xyzw).as_matrix()
    return np.hstack([Rm[:, 0], Rm[:, 1]]).astype(np.float32)  # [r00,r10,r20,r01,r11,r21]

# ───────── image ─────────
img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

# ───────── CSV → 32-D state vec for inference ─────────
df   = pd.read_csv(CSV_PATH)
row0 = df.iloc[idx]

left_pos  = row0[["left_pos_x","left_pos_y","left_pos_z"]].to_numpy(np.float32)
right_pos = row0[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy(np.float32)

left_q   = row0[["left_ori_x","left_ori_y","left_ori_z","left_ori_w"]].to_numpy(np.float32)
right_q  = row0[["right_ori_x","right_ori_y","right_ori_z","right_ori_w"]].to_numpy(np.float32)
left_r6  = quat_xyzw_to_rot6d(left_q)
right_r6 = quat_xyzw_to_rot6d(right_q)

left_pad  = np.array([0.0], dtype=np.float32)
right_pad = np.array([0.0], dtype=np.float32)

hands = row0[
    ["left_hand_0","left_hand_1","left_hand_2","left_hand_3","left_hand_4","left_hand_5",
     "right_hand_0","right_hand_1","right_hand_2","right_hand_3","right_hand_4","right_hand_5"]
].to_numpy(np.float32)

state_vec = np.concatenate([left_pos, left_r6, left_pad, right_pos, right_r6, right_pad, hands], dtype=np.float32)
assert state_vec.shape[0] == ACTION_DIM, state_vec.shape

# ───────── example dict ─────────
example = {
    "state": jnp.asarray(state_vec),
    "image": jnp.asarray(img.astype(np.uint8)),
    "prompt": PROMPT,
}

# ───────── load policy & infer ─────────
conf = cfg.get_config("pi0_galaxea")
# Optional: ensure action_dim matches (usually baked into checkpoint)
# conf.model.action_dim = ACTION_DIM

ckpt_dir = download.maybe_download(model_path)
policy   = policy_config.create_trained_policy(conf, ckpt_dir)

pred = np.asarray(policy.infer(example)["actions"])  # shape (T, 32)

# ───────── ground-truth wrist trajectories from CSV ─────────
# We only need positions for plotting; CSV already has pos columns.
gt_rows = df.iloc[idx : idx + len(pred)]
traj_gt_L = gt_rows[["left_pos_x","left_pos_y","left_pos_z"]].to_numpy(np.float32)        # (T,3)
traj_gt_R = gt_rows[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy(np.float32)     # (T,3)

# predicted wrist xyz slices under new layout
traj_pred_L = pred[:, 0:3]      # left pos
traj_pred_R = pred[:, 10:13]    # right pos (after L pos(3)+L rot6d(6)+pad(1) = 10)

# ───────── plot ─────────
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")

ax.plot(*traj_pred_L.T, label="pred left wrist")
ax.plot(*traj_gt_L.T,   label="GT left wrist",   linestyle="--")

ax.plot(*traj_pred_R.T, label="pred right wrist")
ax.plot(*traj_gt_R.T,   label="GT right wrist",  linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Predicted vs Ground-Truth Wrist Trajectories (32-D layout)")
ax.legend()
ax.set_box_aspect([1,1,1])
fig.savefig("wrist_traj_pred_vs_gt.png", dpi=300, bbox_inches="tight")
plt.close(fig)
