from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union
import os
import sys

import torch
from peft.peft_model import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


def _bootstrap_src_path() -> Optional[Path]:
    """Ensure <repo>/src is on sys.path even when harness files are copied elsewhere."""
    candidates = []
    submit_dir = os.getenv("SLURM_SUBMIT_DIR")
    if submit_dir:
        candidates.append(Path(submit_dir).resolve())
    candidates.append(Path(__file__).resolve())
    src_dir: Optional[Path] = None
    for candidate in candidates:
        for base in [candidate] + list(candidate.parents):
            maybe_src = base if base.name == "src" else base / "src"
            if maybe_src.is_dir():
                src_dir = maybe_src
                break
        if src_dir:
            break
    if src_dir:
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    return src_dir


_bootstrap_src_path()

from helper.repo_paths import ensure_modeling_on_sys_path  # noqa: E402

ensure_modeling_on_sys_path()

try:
    from sae import load_sae as _load_project_sae  # noqa: E402
except Exception:  # pragma: no cover - optional dependency
    _load_project_sae = None


def _maybe_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"", "none"}:
            return None
    return int(value)


def _maybe_float(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    return float(value)


def _maybe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalise_space(value: Any) -> str:
    if value is None:
        return "dense"
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"dense", "hidden", "non_sparse", "nonsparse"}:
            return "dense"
        if normalised in {"sparse", "sae"}:
            return "sparse"
    raise ValueError(
        f"Unsupported steering space '{value}'. Expected 'dense' or 'sparse'."
    )


@contextmanager
def steer(
    model: Union[PreTrainedModel, PeftModel], hook_to_steer: dict[str, Callable]
) -> Generator[None, Any, None]:
    """
    Context manager that temporarily hooks models and steers them.

    Args:
        model: The transformer model to hook
        hook_to_steer: Dictionary mapping hookpoints to steering functions

    Yields:
        None
    """

    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor):
            # If output is a tuple (like in some transformer layers), take first element
            if isinstance(output, tuple):
                output = (hook_to_steer[hookpoint](output[0]), *output[1:])  # type: ignore
            else:
                output = hook_to_steer[hookpoint](output)

            return output

        return hook_fn

    handles = []
    hookpoints = list(hook_to_steer.keys())

    for name, module in model.base_model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_hook(name))
            handles.append(handle)

    if len(handles) != len(hookpoints):
        raise ValueError(f"Not all hookpoints could be resolved: {hookpoints}")

    try:
        yield None
    finally:
        for handle in handles:
            handle.remove()


@register_model("steered")
class SteeredHF(HFLM):
    hook_to_steer: dict[str, Callable]

    def __init__(
        self,
        pretrained: str,
        steer_path: str,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        HFLM with a steered forward pass.

        To load steering vectors directly, provide the path to a pytorch (.pt) file with content in the following format:

        {
            hookpoint: {
                "steering_vector": <torch.Tensor>,
                "steering_coefficient": <float>,
                "action": <Literal["add", "clamp"]>,
                "bias": <torch.Tensor | None>,
                "head_index": <int | None>,
            },
            ...
        }

        To derive steering vectors from a sparse model loadable with sparsify or sae_lens,
        provide the path to a CSV file with the following columns (example rows are provided below):

        loader,action,sparse_model,hookpoint,feature_index,steering_coefficient,head_index,sae_id,description,
        sparsify,add,EleutherAI/sae-pythia-70m-32k,layers.3,30,10.0,,,,
        sae_lens,add,gemma-scope-2b-pt-res-canonical,layers.20,12082,240.0,,layer_20/width_16k/canonical,increase dogs,
        """
        steering_layer = _maybe_optional_int(kwargs.pop("steering_layer", None))
        steering_hookpoint = kwargs.pop("steering_hookpoint", None)
        head_index = _maybe_optional_int(kwargs.pop("head_index", None))
        steering_multiplier = _maybe_float(kwargs.pop("steering_multiplier", 1.0))
        steering_space = _normalise_space(kwargs.pop("steering_space", "dense"))
        steering_relu = _maybe_bool(kwargs.pop("steering_relu", True))
        sae_repo = kwargs.pop("sae_repo", "google/gemma-scope-2b-pt-res")
        sae_vector_size = kwargs.pop("sae_vector_size", "65k")
        sae_avg_idx = kwargs.pop("sae_avg_idx", "68")
        sae_source = kwargs.pop("sae_source", None)
        if isinstance(sae_source, str) and sae_source.strip().lower() in {"", "none"}:
            sae_source = None
        llama_sae_site = kwargs.pop("llama_sae_site", "R")
        llama_sae_expansion = _maybe_optional_int(kwargs.pop("llama_sae_expansion", 8))
        if llama_sae_expansion is None:
            llama_sae_expansion = 8
        llama_sae_cache_dir = kwargs.pop("llama_sae_cache_dir", None)
        if isinstance(llama_sae_cache_dir, str) and llama_sae_cache_dir.strip().lower() in {
            "",
            "none",
        }:
            llama_sae_cache_dir = None

        steering_meta = {
            "layer": steering_layer,
            "hookpoint": steering_hookpoint,
            "head_index": head_index,
            "space": steering_space,
            "multiplier": steering_multiplier,
            "relu": steering_relu,
            "sae_repo": sae_repo,
            "sae_vector_size": sae_vector_size,
            "sae_avg_idx": sae_avg_idx,
            "sae_source": sae_source,
            "llama_sae_site": llama_sae_site,
            "llama_sae_expansion": llama_sae_expansion,
            "llama_sae_cache_dir": llama_sae_cache_dir,
        }

        super().__init__(pretrained=pretrained, device=device, **kwargs)

        steer_config = self._load_steer_config(
            steer_path=steer_path, steering_meta=steering_meta
        )

        hook_to_steer = {}
        for hookpoint, steer_info in steer_config.items():
            action = steer_info["action"]
            steering_vector = steer_info["steering_vector"].to(self.device)
            if action != "sparse_add":
                steering_vector = steering_vector.to(self.model.dtype)
            steering_coefficient = float(steer_info.get("steering_coefficient", 1.0))
            head_index = steer_info.get("head_index", None)
            bias = steer_info.get("bias", None)
            if bias is not None:
                bias = bias.to(self.device).to(self.model.dtype)

            if action == "add":
                # Steer the model by adding a multiple of a steering vector to all sequence positions.
                assert bias is None, "Bias is not supported for the `add` action."
                hook_to_steer[hookpoint] = partial(
                    self.add,
                    vector=steering_vector * steering_coefficient,
                    head_index=head_index,
                )
            elif action == "clamp":
                # Steer the model by clamping the activations to a value in the direction of the steering vector.
                hook_to_steer[hookpoint] = partial(
                    self.clamp,
                    direction=steering_vector / torch.norm(steering_vector),
                    value=steering_coefficient,
                    bias=bias,
                    head_index=head_index,
                )
            elif action == "sparse_add":
                sae_model = steer_info.get("sae")
                if sae_model is None:
                    raise ValueError("Sparse steering requires an SAE module.")
                sae_model = sae_model.to(self.device)
                sae_param = next(iter(sae_model.parameters()), None)
                sae_dtype = (
                    sae_param.dtype if sae_param is not None else self.model.dtype
                )
                steering_vector = steering_vector.to(sae_dtype)
                hook_to_steer[hookpoint] = partial(
                    self.sparse_add,
                    vector=steering_vector * steering_coefficient,
                    sae=sae_model,
                    apply_relu=_maybe_bool(steer_info.get("steering_relu"), True),
                )
            else:
                raise ValueError(f"Unknown hook type: {action}")

        self.hook_to_steer = hook_to_steer

    def _load_steer_config(
        self, steer_path: str, steering_meta: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        if steer_path.endswith(".pt") or steer_path.endswith(".pth"):
            with open(steer_path, "rb") as f:
                loaded = torch.load(
                    f,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )

            if self._is_steer_mapping(loaded):
                return loaded  # type: ignore[return-value]
            if torch.is_tensor(loaded):
                hookpoint = self._resolve_hookpoint(
                    steering_meta.get("layer"), steering_meta.get("hookpoint")
                )
                action = (
                    "sparse_add" if steering_meta["space"] == "sparse" else "add"
                )
                return self._build_tensor_config(
                    hookpoint=hookpoint,
                    vector=loaded,
                    action=action,
                    steering_meta=steering_meta,
                )
            raise ValueError(
                f"Unsupported steering checkpoint format at '{steer_path}'. "
                "Expected a dict config or raw tensor."
            )
        if steer_path.endswith(".csv"):
            return self.derive_steer_config(steer_path)
        raise ValueError(f"Unknown steer file type: {steer_path}")

    @staticmethod
    def _is_steer_mapping(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        return all(isinstance(v, dict) and "action" in v for v in obj.values())

    def _build_tensor_config(
        self,
        hookpoint: str,
        vector: Tensor,
        action: str,
        steering_meta: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        vec = vector.detach().clone()
        if vec.dim() > 1:
            vec = vec.reshape(-1)

        config: dict[str, dict[str, Any]] = {
            hookpoint: {
                "action": action,
                "steering_vector": vec,
                "steering_coefficient": steering_meta["multiplier"],
            }
        }
        head_index = steering_meta.get("head_index")
        if head_index is not None and action != "sparse_add":
            config[hookpoint]["head_index"] = head_index

        if action == "sparse_add":
            if _load_project_sae is None:
                raise ImportError(
                    "Sparse steering requires modeling/yapo/sae.py to be available."
                )
            layer_idx = steering_meta.get("layer")
            if layer_idx is None:
                raise ValueError(
                    "steering_layer must be provided when steering_space='sparse'."
                )
            sae = _load_project_sae(
                model_path=steering_meta["sae_repo"],
                layer_index=int(layer_idx),
                sae_vector_size=steering_meta["sae_vector_size"],
                avg_idx=steering_meta["sae_avg_idx"],
                dtype=getattr(self.model, "dtype", None),
                sae_source=steering_meta["sae_source"],
                llama_site=steering_meta["llama_sae_site"],
                llama_expansion=int(steering_meta["llama_sae_expansion"]),
                llama_cache_dir=steering_meta["llama_sae_cache_dir"],
            )
            sae = sae.to(self.device)
            sae.eval()
            config[hookpoint]["sae"] = sae
            config[hookpoint]["steering_relu"] = steering_meta["relu"]

        return config

    def _resolve_hookpoint(
        self, layer: Optional[int], explicit_hookpoint: Optional[str]
    ) -> str:
        if explicit_hookpoint:
            return str(explicit_hookpoint)
        if layer is None:
            raise ValueError(
                "steering_layer or steering_hookpoint must be provided when using a raw tensor steer file."
            )

        names = self._get_hookpoint_names()
        layer_str = str(int(layer))
        candidates = [
            f"model.layers.{layer_str}",
            f"layers.{layer_str}",
            f"transformer.h.{layer_str}",
            f"transformer.layers.{layer_str}",
            f"gpt_neox.layers.{layer_str}",
            f"model.decoder.layers.{layer_str}",
            f"decoder.layers.{layer_str}",
            f"h.{layer_str}",
            layer_str,
        ]
        for candidate in candidates:
            if candidate in names:
                return candidate
        suffix = f".{layer_str}"
        for name in names:
            if name.endswith(suffix):
                return name
        raise ValueError(
            f"Unable to resolve hookpoint for layer {layer_str}. "
            "Specify --model_args steering_hookpoint=<module.path>."
        )

    def _get_hookpoint_names(self) -> set[str]:
        if not hasattr(self, "_hookpoint_names"):
            self._hookpoint_names = {
                name for name, _ in self.model.base_model.named_modules() if name
            }
        return self._hookpoint_names

    @classmethod
    def derive_steer_config(cls, steer_path: str):
        """Derive a dictionary of steering vectors from sparse model(/s) specified in a CSV file."""
        import pandas as pd

        df = pd.read_csv(steer_path)
        steer_data: dict[str, dict[str, Any]] = {}

        if any(df["loader"] == "sparsify"):
            from sparsify import SparseCoder
        if any(df["loader"] == "sae_lens"):
            from sae_lens import SAE

            sae_cache = {}

            def load_from_sae_lens(sae_release: str, sae_id: str):
                cache_key = (sae_release, sae_id)
                if cache_key not in sae_cache:
                    sae_cache[cache_key] = SAE.from_pretrained(sae_release, sae_id)[0]

                return sae_cache[cache_key]

        for _, row in df.iterrows():
            action = row.get("action", "add")
            sparse_name = row["sparse_model"]
            hookpoint = row["hookpoint"]
            feature_index = int(row["feature_index"])
            steering_coefficient = float(row["steering_coefficient"])
            loader = row.get("loader", "sparsify")

            if loader == "sparsify":
                name_path = Path(sparse_name)

                sparse_coder = (
                    SparseCoder.load_from_disk(name_path / hookpoint)
                    if name_path.exists()
                    else SparseCoder.load_from_hub(sparse_name, hookpoint)
                )
                assert sparse_coder.W_dec is not None

                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec

            elif loader == "sae_lens":
                sparse_coder = load_from_sae_lens(
                    sae_release=sparse_name, sae_id=row["sae_id"]
                )
                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec
                if hookpoint == "" or pd.isna(hookpoint):
                    hookpoint = sparse_coder.cfg.hook_name
            else:
                raise ValueError(f"Unknown loader: {loader}")

            steer_data[hookpoint] = {
                "action": action,
                "steering_coefficient": steering_coefficient,
                "steering_vector": steering_vector,
                "bias": bias,
            }

        return steer_data

    @classmethod
    def add(
        cls,
        acts: Tensor,
        vector: Tensor,
        head_index: Optional[int],
    ):
        """Adds the given vector to the activations.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            vector (Tensor): A vector to add of shape [features]
            head_index (int | None): Optional attention head index to add to
        """
        if head_index is not None:
            acts[:, :, head_index, :] = acts[:, :, head_index, :] + vector
        else:
            acts = acts + vector

        return acts

    @classmethod
    def sparse_add(
        cls,
        acts: Tensor,
        vector: Tensor,
        sae: nn.Module,
        apply_relu: bool = True,
    ):
        """Applies sparse steering by encoding acts via an SAE, injecting a vector in
        sparse space, then decoding back with a reconstruction correction."""

        original_dtype = acts.dtype
        hidden = acts
        original_shape = hidden.shape

        sae_param = next(iter(sae.parameters()), None)
        sae_dtype = sae_param.dtype if sae_param is not None else hidden.dtype

        flat_hidden = hidden.reshape(-1, hidden.size(-1)).to(dtype=sae_dtype)
        sparse_codes = sae.encode(flat_hidden)

        steer_vec = vector
        if steer_vec.dim() == 1:
            steer_vec = steer_vec.view(1, -1)
        steer_vec = steer_vec.to(device=sparse_codes.device, dtype=sparse_codes.dtype)
        steered_sparse = sparse_codes + steer_vec
        if apply_relu:
            steered_sparse = torch.relu(steered_sparse)

        steered_hidden = sae.decode(steered_sparse)
        recon = sae.decode(sparse_codes)
        delta = flat_hidden - recon
        final_hidden = steered_hidden + delta

        return final_hidden.view(original_shape).to(dtype=original_dtype)

    @classmethod
    def clamp(
        cls,
        acts: Tensor,
        direction: Tensor,
        value: float,
        head_index: Optional[int],
        bias: Optional[Tensor] = None,
    ):
        """Clamps the activations to a given value in a specified direction. The direction
        must be a unit vector.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            direction (Tensor): A direction to clamp of shape [features]
            value (float): Value to clamp the direction to
            head_index (int | None): Optional attention head index to clamp
            bias (Tensor | None): Optional bias to add to the activations

        Returns:
            Tensor: The modified activations with the specified direction clamped
        """
        if bias is not None:
            acts = acts - bias

        if head_index is not None:
            x = acts[:, :, head_index, :]
            proj = (x * direction).sum(dim=-1, keepdim=True)
            assert proj == acts @ direction

            clamped = acts.clone()
            clamped[:, :, head_index, :] = x + direction * (value - proj)
        else:
            proj = torch.sum(acts * direction, dim=-1, keepdim=True)
            clamped = acts + direction * (value - proj)

        if bias is not None:
            return clamped + bias

        return clamped

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            with steer(self.model, self.hook_to_steer):
                return self.model.forward(*args, **kwargs)

    def _model_call(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_generate(*args, **kwargs)
