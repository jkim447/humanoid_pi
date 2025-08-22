# Usage (7/31/2025, for pickup can)
# jwbkim@iris8:/iris/projects/humanoid/openpi$ uv run examples/libero/convert_our_data_to_lerobot.py --data_dir /iris/projects/humanoid/dataset/recordstart_2025-07-09_22-26-20


# notes regarding policy actions, from https://github.com/Physical-Intelligence/openpi/discussions/302
# I can, but please keep in mind that pi0-base is not intended to be used zero-shot. The correctness of the actions depends on the postprocessing code being exactly correct, and we have not provided postprocessing code (or even norm stats) for pi0-base. We will likely not release these because they don't really make sense outside our internal robot setups.

# Even if the postprocessing is correct, we are still far from our models achieving zero-shot robot, environment, and task generalization.

# That said, here are the nitty-gritty details:

# joint position deltas: no change
# end-effector control in a fixed global reference frame: "<control_mode> end effector </control_mode> {language_prompt}"
# end-effector control in end-effector reference frame (i.e., deltas): "<control_mode> end effector cam frame </control_mode> {language_prompt}"
# End-effector poses are encoded using XYZ position and Rot6D rotation (first two columns of the rotation matrix, flattened). The ordering is [x, y, z, *rot6d, gripper], so 9 dimensions total. For biarm setups, this sequence is repeated first for the left arm and then for the right arm. For end-effector deltas, the reference frame is lined up with the wrist camera such that +z is forward, +x is right, and +y is down.

# pi0-fast-base was not trained with additional control modes.



import shutil
import pandas as pd
from pathlib import Path
import cv2
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
from scipy.spatial.transform import Rotation as R

REPO_NAME = "jkim447/pick_up_can_dataset"  # Change to your HF dataset repo
print("saving to folder: ", HF_LEROBOT_HOME)
# assert False
def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    # quat_xyzw: [x, y, z, w] (scalar-last). Scipy expects this ordering.
    rot = R.from_quat(quat_xyzw)            # 3x3 rotation
    Rm = rot.as_matrix()
    # First two columns, flattened [r00, r10, r20, r01, r11, r21]
    return np.hstack([Rm[:, 0], Rm[:, 1]]).astype(np.float32)

def main(data_dir: str, *, push_to_hub: bool = False):
    data_dir = Path(data_dir)

    # Remove any existing dataset at output path
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Define LeRobot dataset schema
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="galaxea",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {  # full pose + hand joints
                "dtype": "float32",
                "shape": (len(_get_state_names()),),
                "names": _get_state_names(),
            },
            "actions": {  # here I'm just using same as state (you may customize)
                "dtype": "float32",
                "shape": (len(_get_state_names()),),
                "names": _get_state_names(),
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    demo_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    # print(demo_dirs)
    for demo_dir in demo_dirs:
        csv_path = demo_dir / "ee_pos/ee_poses_and_hands.csv"
        img_dir = demo_dir / "left"

        # print("im here")
        # print(csv_path)
        # print(img_dir)
        # print(not csv_path.exists(), not img_dir.exists())
        if not csv_path.exists() or not img_dir.exists():
            continue

        # print("im here1")
        df = pd.read_csv(csv_path)

        # Loop over row index instead of frame_id column
        for idx in range(len(df)):
            img_filename = f"{idx:06d}.jpg"
            img_path = img_dir / img_filename
            # print(img_path)
            if not img_path.exists():
                continue

            # Load & resize image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            # # Create state/action arrays from CSV row (excluding frame_id if it exists)
            # row = df.iloc[idx]
            # if "frame_id" in row.index:
            #     row = row.drop(labels=["frame_id"])
            # state = row.to_numpy(dtype=np.float32)

            row = df.iloc[idx]

            # positions
            left_pos  = row[["left_pos_x","left_pos_y","left_pos_z"]].to_numpy(dtype=np.float32)
            right_pos = row[["right_pos_x","right_pos_y","right_pos_z"]].to_numpy(dtype=np.float32)

            # quaternions -> 6D
            left_quat_xyzw  = row[["left_ori_x","left_ori_y","left_ori_z","left_ori_w"]].to_numpy(dtype=np.float32)
            right_quat_xyzw = row[["right_ori_x","right_ori_y","right_ori_z","right_ori_w"]].to_numpy(dtype=np.float32)
            left_rot6d  = quat_xyzw_to_rot6d(left_quat_xyzw)
            right_rot6d = quat_xyzw_to_rot6d(right_quat_xyzw)

            # padding scalars
            left_pad  = np.array([0.0], dtype=np.float32)
            right_pad = np.array([0.0], dtype=np.float32)

            # hand joints (6 + 6)
            hands = row[
                [
                    "left_hand_0","left_hand_1","left_hand_2","left_hand_3","left_hand_4","left_hand_5",
                    "right_hand_0","right_hand_1","right_hand_2","right_hand_3","right_hand_4","right_hand_5",
                ]
            ].to_numpy(dtype=np.float32)

            state = np.concatenate(
                [left_pos, left_rot6d, left_pad, right_pos, right_rot6d, right_pad, hands],
                dtype=np.float32,
            )

            # print(state.shape)
            # print(state)
            # assert False

            dataset.add_frame(
                {
                    "image": img,
                    "state": state,
                    "actions": state,  # adjust if actions are different
                    "task": "pick up can and move it to the center"
                }
            )
        print("saving episode")
        dataset.save_episode()

# if push_to_hub:
#     dataset.push_to_hub(
#         tags=["custom", "robot"],
#         private=False,
#         push_videos=False,
#         license="apache-2.0",
#     )



# def _get_state_names():
#     return [
#         "left_pos_x", "left_pos_y", "left_pos_z",
#         "left_ori_x", "left_ori_y", "left_ori_z", "left_ori_w",
#         "right_pos_x", "right_pos_y", "right_pos_z",
#         "right_ori_x", "right_ori_y", "right_ori_z", "right_ori_w",
#         "left_hand_0", "left_hand_1", "left_hand_2", "left_hand_3", "left_hand_4", "left_hand_5",
#         "right_hand_0", "right_hand_1", "right_hand_2", "right_hand_3", "right_hand_4", "right_hand_5",
#     ]

def _get_state_names():
    return [
        # left
        "left_pos_x","left_pos_y","left_pos_z",
        "left_rot6d_0","left_rot6d_1","left_rot6d_2","left_rot6d_3","left_rot6d_4","left_rot6d_5",
        "left_pad",
        # right
        "right_pos_x","right_pos_y","right_pos_z",
        "right_rot6d_0","right_rot6d_1","right_rot6d_2","right_rot6d_3","right_rot6d_4","right_rot6d_5",
        "right_pad",
        # hands (6 + 6)
        "left_hand_0","left_hand_1","left_hand_2","left_hand_3","left_hand_4","left_hand_5",
        "right_hand_0","right_hand_1","right_hand_2","right_hand_3","right_hand_4","right_hand_5",
    ]


if __name__ == "__main__":
    tyro.cli(main)
