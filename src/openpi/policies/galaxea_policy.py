import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# TODO: CHANGE ME AS NEEDED!
action_dim = 48

def make_galaxea_example() -> dict:
    """Creates a random input example for the Galaxea policy."""
    return {
        # Replace state dimension with your Galaxea proprioceptive dimension if different.
        "state": np.random.rand(action_dim), #
        # Replace with actual Galaxea image sources / shapes.
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # Galaxea prompt (language instruction) if applicable.
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Ensures image is uint8 in (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GalaxeaInputs(transforms.DataTransformFn):
    """
    Converts Galaxea dataset inputs into the expected model format.
    Used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on
    the comments below to route the correct elements of your dataset into the model.
    """

    # Action dimension of the model — used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change unless your model action dimension is different.
    action_dim: int

    # Model type determines certain preprocessing behaviors.
    # Keep as is unless you change model architecture.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad proprioceptive input to match action dimension.
        # Change key if your Galaxea proprioceptive state key is different.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Convert images to uint8 (H, W, C). Adjust keys if Galaxea stores images differently.
        base_image = _parse_image(data["image"])
        # wrist_image = _parse_image(data["observation/wrist_image"])

        # Create model input dict — keep structure identical to Libero for compatibility.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad missing views with zeros if not available.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_, # TODO: hardcode to false?
                # Mask missing views as False if masking is enabled
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_, # TODO: hardcode to false?
            },
        }

        # Include actions if present (training only).
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Include prompt (language instruction) if present.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GalaxeaOutputs(transforms.DataTransformFn):
    """
    Converts model outputs into Galaxea dataset-specific format.
    Used for inference only.
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Return only the actual action dimensions, excluding padding.
        # Change `7` to your Galaxea action dimension if different.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])} 