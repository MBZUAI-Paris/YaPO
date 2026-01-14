from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

DEFAULT_SAE_SOURCE = "gemma_scope"

class JumpReLUSAE(nn.Module):
    """ 
    Simple Autoencoder with JumpReLU non-linearity.
    Implementation by Google: https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp#scrollTo=8wy7DSTaRc90
    """
    def __init__(self, d_model, d_sae):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon


def load_sae(
    model_path: str = "google/gemma-scope-2b-pt-res",
    layer_index: int = 17,
    sae_vector_size: str = "65k",
    avg_idx: str = "68",
    dtype: torch.dtype | None = None,
    sae_source: str | None = None,
    llama_site: str = "R",
    llama_expansion: int = 8,
    llama_cache_dir: str | None = None,
):
    """
    Load a pretrained SAE from Hugging Face.

    Parameters
    ----------
    model_path:
        Gemma-Scope repository ID (when ``sae_source`` is ``gemma_scope``) or an
        optional override for Llama-Scope repositories.
    layer_index:
        Transformer layer the SAE was trained on.
    sae_vector_size / avg_idx:
        Gemma-Scope specific selectors retained for backward compatibility.
    dtype:
        Optional torch dtype to cast weights to.
    sae_source:
        Which SAE family to load. Supported values: ``gemma_scope`` (default)
        and ``llama_scope``. If omitted, the source is inferred from
        ``model_path`` (strings containing ``llama`` map to ``llama_scope``).
    llama_site / llama_expansion / llama_cache_dir:
        Parameters required when ``sae_source='llama_scope'`` matching the
        Llama-Scope naming convention (site ∈ {R,A,M,TC}, expansion ∈ {8,32}).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = _normalize_source(model_path=model_path, sae_source=sae_source)

    if source == "llama_scope":
        return _load_llama_scope_sae(
            layer_index=layer_index,
            dtype=dtype,
            device=device,
            repo_override=model_path,
            site=llama_site,
            expansion=llama_expansion,
            cache_dir=llama_cache_dir,
        )

    return _load_gemma_scope_sae(
        model_path=model_path,
        layer_index=layer_index,
        sae_vector_size=sae_vector_size,
        avg_idx=avg_idx,
        dtype=dtype,
        device=device,
    )


def _normalize_source(model_path: str, sae_source: str | None) -> str:
    if sae_source:
        return sae_source.strip().lower()
    if "llama" in (model_path or "").lower():
        return "llama_scope"
    return DEFAULT_SAE_SOURCE


def _load_gemma_scope_sae(
    model_path: str,
    layer_index: int,
    sae_vector_size: str,
    avg_idx: str,
    dtype: torch.dtype | None,
    device: torch.device,
):
    sae_id = f"layer_{layer_index}/width_{sae_vector_size}/average_l0_{avg_idx}/params.npz"

    path_to_params = hf_hub_download(
        repo_id=model_path,
        filename=sae_id,
        force_download=False,
    )

    params = np.load(path_to_params, allow_pickle=False)
    pt_params = {}
    for k, v in params.items():
        t_param = torch.from_numpy(v).to(device)
        if dtype is not None:
            t_param = t_param.to(dtype)
        pt_params[k] = t_param

    print(f"Loaded Gemma-Scope SAE from {path_to_params}")
    shapes = {k: v.shape for k, v in pt_params.items()}
    print(f"SAE params shapes: {shapes}")

    sae = JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1])
    sae.load_state_dict(pt_params)
    return sae.to(device)


def _load_llama_scope_sae(
    layer_index: int,
    dtype: torch.dtype | None,
    device: torch.device,
    repo_override: str | None,
    site: str,
    expansion: int,
    cache_dir: str | None,
):
    try:
        from .llama_sae import TopKSparseAutoEncoder, download_sae
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Llama-Scope support requires modeling/yapo/sae/llama_sae.py "
            "and its dependencies (huggingface_hub + safetensors)."
        ) from exc

    repo_id = repo_override if repo_override and "gemma" not in repo_override.lower() else None
    weights_path, hyperparams = download_sae(
        layer=layer_index,
        site=site,
        expansion=int(expansion),
        cache_dir=cache_dir,
        repo_id=repo_id,
    )
    sae = TopKSparseAutoEncoder(weights_path, hyperparams, device=str(device))
    print(
        f"Loaded Llama-Scope SAE from {weights_path} "
        f"(layer={layer_index}, site={site}, expansion={expansion}x)"
    )
    if dtype is not None:
        sae = sae.to(dtype=dtype)
    return sae.to(device)
