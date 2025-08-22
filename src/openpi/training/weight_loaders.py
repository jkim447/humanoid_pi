import dataclasses
import logging
import os, re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        # Flatten checkpoint params
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        # If your checkpoint stores everything under "params/...", strip it to match model keys
        if flat_loaded and all(isinstance(k, str) and k.startswith("params/") for k in flat_loaded):
            flat_loaded = {k[len("params/"):]: v for k, v in flat_loaded.items()}

        # Drop exactly the offending keys. Use a robust regex to handle optional prefixes.
        drop_re = re.compile(
            r"(?:.*/)?(?:(?:action_(?:in|out)_proj/(?:kernel|bias))|(state_proj/kernel))$"
        )

        dropped = []
        for k in list(flat_loaded.keys()):
            if drop_re.fullmatch(k):
                dropped.append((k, getattr(flat_loaded[k], "shape", None)))
                del flat_loaded[k]

        if dropped:
            print("[CheckpointWeightLoader] Dropping keys so they re-init:")
            for k, shp in dropped:
                print("  DROP", k, "shape:", shp)

        # Rebuild nested dict
        loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")

        # original
        # return _merge_params(loaded_params, params, missing_regex=".*lora.*")

        # TODO: missing_regex of ".*" might be too general and remiss
        return _merge_params(loaded_params, params, missing_regex=".*") 


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/") # NOTE: loaded params do not contain the action projection layers, since we surgically removed them

    # Take weights whose keys exist AND shapes match; cast to ref dtype.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            if getattr(v, "shape", None) == getattr(ref, "shape", None):
                result[k] = v.astype(getattr(ref, "dtype", getattr(v, "dtype", None)))
            # else: skip; will be backfilled below if allowed by regex

    # Backfill missing/skipped keys according to policy.
    pattern = re.compile(missing_regex)
    for k, ref in flat_ref.items():
        if k not in result and pattern.fullmatch(k):
            result[k] = ref

    return flax.traverse_util.unflatten_dict(result, sep="/")


# def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
#     flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
#     flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

#     def _shape(x):
#         # works for np.ndarray, jax arrays, and ShapeDtypeStruct
#         return tuple(getattr(x, "shape", ())) if hasattr(x, "shape") else None

#     def _sample(d, n=30):
#         ks = sorted(d.keys())
#         return ks[:n], max(0, len(ks) - n)

#     # Print small samples
#     ref_keys = set(flat_ref.keys())
#     ld_keys  = set(flat_loaded.keys())

#     only_in_ref   = sorted(ref_keys - ld_keys)
#     only_in_ckpt  = sorted(ld_keys  - ref_keys)
#     common        = sorted(ref_keys & ld_keys)

#     mismatched = []
#     for k in common:
#         if _shape(flat_ref[k]) != _shape(flat_loaded[k]):
#             mismatched.append((k, _shape(flat_loaded[k]), _shape(flat_ref[k])))

#     print("=== _merge_params DEBUG ===")
#     print(f"ref_keys: {len(ref_keys)}  ckpt_keys: {len(ld_keys)}  common: {len(common)}")
#     s, more = _sample({k:None for k in only_in_ref});  print(f"only_in_ref: {len(only_in_ref)}");  [print("  ", k) for k in s];  print(("  ... +%d more" % more) if more else "")
#     s, more = _sample({k:None for k in only_in_ckpt}); print(f"only_in_ckpt: {len(only_in_ckpt)}"); [print("  ", k) for k in s]; print(("  ... +%d more" % more) if more else "")
#     s, more = _sample({k:None for k,_,_ in mismatched}); print(f"mismatched_shapes: {len(mismatched)}")
#     for k, ck, rf in mismatched[:30]:
#         print("  ", k, " ckpt:", ck, " ref:", rf)
#     print("===========================")

#     assert False

#     # Toggle verbose logging with: OPENPI_LOG_WEIGHT_MERGE=1
#     _LOG = os.environ.get("OPENPI_LOG_WEIGHT_MERGE", "0") == "1"

#     loaded_ok = []         # matched + used
#     mismatched = []        # present in ckpt but shape mismatch → skipped
#     extra_in_ckpt = []     # present in ckpt but not in ref
#     backfilled = []        # missing/ skipped → filled from ref
#     result = {}

#     # 1) Take only weights whose keys exist and shapes match.
#     for k, v in flat_loaded.items():
#         if k in flat_ref:
#             ref = flat_ref[k]
#             if getattr(v, "shape", None) == getattr(ref, "shape", None):
#                 result[k] = v.astype(ref.dtype)
#                 if _LOG:
#                     loaded_ok.append((k, v.shape))
#             else:
#                 if _LOG:
#                     mismatched.append((k, getattr(v, "shape", None), getattr(ref, "shape", None)))
#         else:
#             if _LOG:
#                 extra_in_ckpt.append((k, getattr(v, "shape", None)))

#     # 2) Backfill anything missing from the reference model init.
#     #    You can pass missing_regex=".*" from the caller to backfill all.
#     pattern = re.compile(missing_regex)
#     for k, ref in flat_ref.items():
#         if k not in result and pattern.fullmatch(k):
#             result[k] = ref
#             if _LOG:
#                 backfilled.append((k, getattr(ref, "shape", None)))

#     if _LOG:
#         def _head(lst, n=20):  # keep logs short
#             return lst[:n], max(0, len(lst) - n)

#         LO, LO_more = _head(loaded_ok)
#         MM, MM_more = _head(mismatched)
#         BF, BF_more = _head(backfilled)
#         EX, EX_more = _head(extra_in_ckpt)

#         print("[_merge_params] loaded_ok:", len(loaded_ok))
#         for k, shp in LO: print("  LOADED ", k, shp)
#         if LO_more: print(f"  ... +{LO_more} more")

#         print("[_merge_params] mismatched_skipped:", len(mismatched))
#         for k, s_ckpt, s_ref in MM: print("  SKIPPED", k, "ckpt:", s_ckpt, "ref:", s_ref)
#         if MM_more: print(f"  ... +{MM_more} more")

#         print("[_merge_params] backfilled_from_init:", len(backfilled))
#         for k, s in BF: print("  BACKFILL", k, s)
#         if BF_more: print(f"  ... +{BF_more} more")

#         print("[_merge_params] extra_in_checkpoint_not_in_model:", len(extra_in_ckpt))
#         for k, s in EX: print("  EXTRA  ", k, s)
#         if EX_more: print(f"  ... +{EX_more} more")

#     return flax.traverse_util.unflatten_dict(result, sep="/")

# def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
#     flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
#     flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

#     result = {}
#     for k, v in flat_loaded.items():
#         if k in flat_ref:
#             ref = flat_ref[k]
#             # Only take it if shape matches
#             if getattr(v, "shape", None) == getattr(ref, "shape", None):
#                 result[k] = v.astype(ref.dtype)
#             # else: skip; will be backfilled below if allowed

#     # Backfill ANY missing keys with the reference (fresh init) — important!
#     pattern = re.compile(missing_regex)
#     for k in flat_ref.keys():
#         if k not in result and pattern.fullmatch(k):
#             result[k] = flat_ref[k]

#     return flax.traverse_util.unflatten_dict(result, sep="/")

# original
# def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
#     """Merges the loaded parameters with the reference parameters.

#     Args:
#         loaded_params: The parameters to merge.
#         params: The reference parameters.
#         missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

#     Returns:
#         A new dictionary with the merged parameters.
#     """
#     flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
#     flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

#     # First, take all weights that are a subset of the reference weights.
#     result = {}
#     for k, v in flat_loaded.items():
#         if k in flat_ref:
#             result[k] = v.astype(flat_ref[k].dtype)

#     # Then, merge any missing weights as defined by the missing regex.
#     pattern = re.compile(missing_regex)
#     for k in {k for k in flat_ref if pattern.fullmatch(k)}:
#         if k not in result:
#             result[k] = flat_ref[k]

#     return flax.traverse_util.unflatten_dict(result, sep="/")
