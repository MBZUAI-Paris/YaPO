"""Run Llama‑Scope SAEs from Hugging Face
================================================

This module demonstrates how to download and use the sparse auto‑encoders (SAEs)
released with the **Llama‑Scope** project.  SAEs are lightweight linear layers
trained to extract sparse, interpretable features from the hidden states of
large language models.  Each SAE is tied to a specific layer and stream (site)
within the base model and is identified by the naming convention documented
in the Llama‑Scope model card【491458465838228†L68-L72】.  The available
positions (sites) are:

  * **R** – Residual stream (post‑MLP residual)
  * **A** – Attention output
  * **M** – MLP output
  * **TC** – Token classifiers (final residual stream before logits)

Each SAE is trained with either an 8× or 32× expansion factor, yielding
approximately 32 K or 128 K sparse features respectively【491458465838228†L68-L72】.
The code below shows how to locate a particular SAE on Hugging Face,
download its hyperparameters and weight tensor, build a simple Top‑K SAE
module in PyTorch, and apply it to hidden states extracted from the Llama 3.1 8B
base model.  It does **not** require the Language‑Model‑SAEs package, so it can
be run in environments where that repository cannot be installed directly.

By default the implementation applies a **TopK‑ReLU** activation followed by
Top‑K sparsification, as described in Table 1 of the Llama‑Scope paper【338104864896878†L414-L418】.
This activation multiplies the positive feature activations by the 2‑norms of
their corresponding decoder vectors when selecting the top features
【338104864896878†L266-L304】.  Some checkpoints may instead use a JumpReLU variant;
the code automatically detects and uses the appropriate nonlinearity
according to the ``act_fn`` entry of the SAE’s hyperparameters.  If future
checkpoints adopt different activations or normalisation schemes, adjust
the detection logic in :func:`parse_hyperparams` and :class:`TopKSparseAutoEncoder` accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

try:
    # huggingface_hub and safetensors are optional.  Install them via pip
    # if they are not already available in your environment:
    #   pip install huggingface_hub safetensors
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safe_load
except ImportError as exc:  # pragma: no cover - safe import
    raise ImportError(
        "This script requires the huggingface_hub and safetensors packages. "
        "Install them with `pip install huggingface_hub safetensors`."
    ) from exc


@dataclass
class SaeHyperParams:
    """Hyper‑parameters read from the SAE's hyperparams.json.

    Parameters
    ----------
    d_model: int
        Hidden size of the base model.
    expansion_factor: int
        Expansion factor (8 or 32).  The number of sparse features is
        ``d_model * expansion_factor``.
    top_k: int
        Number of non‑zero features to retain per token.  If None, all
        activations are kept.
    act_fn: str
        Name of the activation function.  Currently only "jumprelu" is
        supported.
    jump_threshold: float
        Threshold used by JumpReLU.  Activations below this threshold
        are zeroed out.
    hook_point_in: str
        Name of the hook from which hidden states should be extracted.
    hook_point_out: str
        Name of the hook into which reconstructions should be injected.
    use_decoder_bias: bool
        Whether a decoder bias is present.  If true, the bias will be added
        during reconstruction.
    device: str
        Device on which the SAE was trained.  Included for completeness but
        not used at runtime.
    """

    d_model: int
    expansion_factor: int
    top_k: int
    act_fn: str
    jump_threshold: float
    hook_point_in: str
    hook_point_out: str
    use_decoder_bias: bool
    device: str


def parse_hyperparams(path: str) -> SaeHyperParams:
    """Parse a hyperparams.json file into an SaeHyperParams instance.

    Parameters
    ----------
    path : str
        Local path to the ``hyperparams.json`` file downloaded from Hugging Face.

    Returns
    -------
    SaeHyperParams
        A dataclass containing the relevant hyperparameters.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SaeHyperParams(
        d_model=data["d_model"],
        expansion_factor=data["expansion_factor"],
        top_k=data.get("top_k", None),
        # Some Llama‑Scope checkpoints record the activation name as
        # ``topk_relu`` or ``jumprelu``.  Normalise to lowercase for
        # downstream dispatch.  Default to ``topk_relu`` if unspecified,
        # following Table 1 of the Llama‑Scope paper【338104864896878†L414-L418】.
        act_fn=data.get("act_fn", "topk_relu").lower(),
        jump_threshold=data.get("jump_relu_threshold", 0.0),
        hook_point_in=data.get("hook_point_in", ""),
        hook_point_out=data.get("hook_point_out", ""),
        use_decoder_bias=data.get("use_decoder_bias", False),
        device=data.get("device", "cpu"),
    )


def jumprelu(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """JumpReLU activation used by Llama‑Scope SAEs.

    Any activation values below ``threshold`` are set to zero.  Values
    above the threshold are linearly offset by the threshold.
    """
    return F.relu(x - threshold)


def apply_topk(values: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the top‑``k`` values and corresponding indices along the last axis.

    The returned ``sparse`` tensor has the same shape as ``values`` but all
    non‑top‑k entries are zeroed out.  The indices tensor stores the
    positions of the top‑k features for each row.
    """
    # values: (batch, n_features)
    if k is None or k >= values.size(-1):
        # Keep all features if k is not specified.
        indices = torch.arange(values.size(-1), device=values.device).expand(values.size(0), -1)
        return values, indices
    topk_vals, topk_idx = values.topk(k=k, dim=-1)
    # Build a sparse activation matrix with zeros everywhere except the top‑k indices
    sparse = torch.zeros_like(values)
    sparse.scatter_(dim=-1, index=topk_idx, src=topk_vals)
    return sparse, topk_idx


class TopKSparseAutoEncoder(torch.nn.Module):
    """Sparse Auto‑Encoder for Llama‑Scope checkpoints.

    This class loads the encoder and decoder weights from a Hugging Face
    ``final.safetensors`` file and applies the nonlinearity specified in
    the SAE hyperparameters followed by a Top‑K sparsification step.  Two
    nonlinearities are currently supported:

    * **TopK‑ReLU** – the canonical Llama‑Scope activation【338104864896878†L414-L418】,
      which first applies a ReLU to the feature activations and then selects
      the top‑``k`` activations based on the product of the activation
      magnitude and the 2‑norm of the corresponding decoder column
      【338104864896878†L266-L304】.  This ensures that features whose decoder vectors
      have larger norms are proportionally more likely to be chosen.
    * **JumpReLU** – a variant introduced in Templeton et al. (2024) and
      supported for completeness【338104864896878†L325-L347】.  JumpReLU subtracts a
      learned threshold from each activation and applies a standard ReLU;
      sparsification then selects the top‑``k`` of these thresholded
      activations.

    After sparsification, the SAE reconstructs the input hidden states by
    multiplying the sparse codes with the decoder and optionally adding a bias.
    The module operates on flattened token embeddings with shape
    ``(batch, d_model)``.
    """

    def __init__(self, weights_path: str, hyperparams: SaeHyperParams, device: Optional[str] = None) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hyperparams

        # Load the weight tensors.  The weight names were inferred from the
        # released checkpoints – encoder weight, decoder weight and optional bias.
        wt = safe_load(weights_path, device=self.device)
        # Select encoder and decoder tensors.  Some checkpoints store
        # multiple tensors (e.g. ``W_enc``, ``W_dec``, biases, norms).  We
        # choose the first 2‑D tensor whose name contains ``enc`` (for
        # encoder) and ``dec`` (for decoder), falling back to any 2‑D
        # tensor if no name matches.  This heuristic avoids picking a 1‑D
        # bias vector and prevents the IndexError observed when a wrong
        # tensor is selected.
        enc_candidates = []
        dec_candidates = []
        for name, tensor in wt.items():
            if tensor.dim() == 2:
                lname = name.lower()
                if "enc" in lname:
                    enc_candidates.append((name, tensor))
                if "dec" in lname or "decoder" in lname:
                    dec_candidates.append((name, tensor))
        # Fallback: include all 2‑D tensors if no explicit matches
        if not enc_candidates:
            enc_candidates = [(n, t) for n, t in wt.items() if t.dim() == 2]
        if not dec_candidates:
            dec_candidates = [(n, t) for n, t in wt.items() if t.dim() == 2]
        # Choose the first candidate tensor for encoder and decoder.
        enc_name, enc_tensor = enc_candidates[0]
        dec_name, dec_tensor = dec_candidates[0]
        self.W_enc = enc_tensor
        self.W_dec = dec_tensor
        # Orient the encoder and decoder matrices so that the hidden size (d_model)
        # dimension aligns with the appropriate axis.  For the encoder we want
        # shape (d_model, n_features).  For the decoder we want shape
        # (n_features, d_model).  If necessary, transpose the tensors.
        if self.W_enc.dim() == 2 and self.W_enc.shape[0] != self.hp.d_model and self.W_enc.shape[1] == self.hp.d_model:
            self.W_enc = self.W_enc.t().contiguous()
        if self.W_dec.dim() == 2 and self.W_dec.shape[1] != self.hp.d_model and self.W_dec.shape[0] == self.hp.d_model:
            self.W_dec = self.W_dec.t().contiguous()
        # Decoder bias, if present.  Many SAEs omit this term.
        bias_keys = [k for k in wt.keys() if "bias" in k.lower()]
        decoder_bias_tensor = wt[bias_keys[0]] if bias_keys else None

        # JumpReLU threshold (used only when act_fn includes "jump")
        self.threshold = float(self.hp.jump_threshold)

        # Top‑k to use when sparsifying.  Can be overridden at call time.
        self.default_topk = self.hp.top_k

        # Register weights as buffers so they move with the model and are not
        # considered parameters (they are fixed during inference).
        self.register_buffer("encoder", self.W_enc, persistent=False)
        self.register_buffer("decoder", self.W_dec, persistent=False)
        # Pre‑compute the L2 norm of each decoder column (row in our representation).
        # These norms are used when computing the TopK gating metric for TopK‑ReLU
        # SAEs【338104864896878†L266-L304】.
        with torch.no_grad():
            decoder_norms = self.decoder.norm(dim=1)
        self.register_buffer("decoder_norms", decoder_norms, persistent=False)
        if decoder_bias_tensor is not None:
            self.register_buffer("decoder_bias", decoder_bias_tensor, persistent=False)

    def encode(self, hidden: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode hidden states into sparse features.

        Parameters
        ----------
        hidden : torch.Tensor
            A tensor of shape ``(batch, d_model)`` representing the token
            activations from the base model.
        k : int, optional
            Number of features to keep per token.  If not specified, the
            default ``top_k`` from the hyperparameters is used.

        Returns
        -------
        sparse_features : torch.Tensor
            A dense tensor of shape ``(batch, n_features)`` with non‑zero
            activations only at the top‑``k`` indices.
        indices : torch.Tensor
            An integer tensor containing the indices of the top features for each
            token.  When ``k`` is ``None`` or greater than the number of
            features, the full set of indices ``0…F−1`` is returned for each
            batch element.
        activations : torch.Tensor
            The raw feature activations prior to sparsification.  For TopK‑ReLU
            this is simply ``ReLU(W_enc·x)``, whereas for JumpReLU a shift of
            ``threshold`` is applied.
        """
        # Ensure the hidden states have the same dtype as the encoder before
        # multiplication to avoid dtype mismatch errors (e.g. float32 vs bfloat16).
        # See the runtime error reported by users: both operands of ``@`` must
        # have identical dtypes.  Casting hidden to ``self.encoder.dtype``
        # preserves numerical consistency of the SAE weights.
        if hidden.dtype != self.encoder.dtype:
            hidden = hidden.to(self.encoder.dtype)
        # Linear projection to feature space
        feats = hidden @ self.encoder  # (batch, n_features)
        # Determine which activation to apply based on hyperparams.  The
        # ``act_fn`` field is normalised to lowercase during parsing.
        act = self.hp.act_fn
        if act.startswith("jump"):
            # JumpReLU: subtract threshold then apply ReLU【338104864896878†L325-L347】
            activations = jumprelu(feats, self.threshold)
            # Use a float32 copy for the gating metric to ensure topk works on
            # low-precision dtypes such as bfloat16.
            gating_metric = activations.to(torch.float32)
        else:
            # TopK‑ReLU or vanilla ReLU: zero out negative activations and
            # compute a gating metric scaled by the decoder norms【338104864896878†L266-L304】.
            relu_feats = F.relu(feats)
            activations = relu_feats
            # Compute the gating metric in float32 to improve numerical stability.
            gating_metric = (relu_feats.to(torch.float32) * self.decoder_norms.to(torch.float32))
        # Sparsify by selecting top‑k based on the gating metric.  If k is
        # unspecified (None or >= n_features), all activations are kept.
        k_val = k if k is not None else self.default_topk
        # When k_val is None or greater than the number of features, we keep
        # everything.  Construct the indices accordingly.
        if k_val is None or k_val >= gating_metric.size(-1):
            # Expand indices to have shape (batch, n_features)
            idx = torch.arange(gating_metric.size(-1), device=gating_metric.device).expand(gating_metric.size(0), -1)
            sparse = activations
            return sparse, idx, activations
        # Otherwise select the top‑k gating scores for each token
        topk_vals, topk_idx = gating_metric.topk(k=k_val, dim=-1)
        # Build a sparse activation matrix: place the original activations at the selected indices
        sparse = torch.zeros_like(activations)
        # Gather the activations corresponding to the top indices
        top_feats = activations.gather(dim=-1, index=topk_idx)
        sparse.scatter_(dim=-1, index=topk_idx, src=top_feats)
        return sparse, topk_idx, activations

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """Reconstruct hidden states from sparse features.

        Parameters
        ----------
        sparse : torch.Tensor
            Sparse feature activations of shape ``(batch, n_features)``.

        Returns
        -------
        recon : torch.Tensor
            Reconstructed hidden states of shape ``(batch, d_model)``.
        """
        recon = sparse @ self.decoder  # (batch, d_model)
        bias = getattr(self, "decoder_bias", None)
        if bias is not None and self.hp.use_decoder_bias:
            # Ensure bias matches the current compute device/dtype before adding.
            if bias.device != recon.device or bias.dtype != recon.dtype:
                bias = bias.to(device=recon.device, dtype=recon.dtype)
            recon = recon + bias
        return recon

    def forward(self, hidden: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode hidden states in a single call.

        Parameters
        ----------
        hidden : torch.Tensor
            Hidden states of shape ``(batch, d_model)``.
        k : int, optional
            Number of features to keep.  Overrides the default top‑k.

        Returns
        -------
        recon : torch.Tensor
            Reconstruction of the hidden states.
        sparse : torch.Tensor
            Sparse feature activations.
        indices : torch.Tensor
            Indices of the top features.
        activations : torch.Tensor
            Raw feature activations before sparsification.
        """
        sparse, indices, activations = self.encode(hidden, k)
        recon = self.decode(sparse)
        return recon, sparse, indices, activations


def resolve_sae_repo(site: str, expansion: int) -> str:
    """Return the Hugging Face repository ID for a given site and expansion.

    The Llama‑Scope checkpoints are organised so that all SAEs for a given site
    and expansion live under one repository, e.g. ``fnlp/Llama3_1-8B-Base-LXR-8x``.

    Parameters
    ----------
    site : str
        One of "R", "A", "M" or "TC".
    expansion : int
        Expansion factor (8 or 32).

    Returns
    -------
    str
        The repository ID on Hugging Face.
    """
    site = site.upper()
    if site not in {"R", "A", "M", "TC"}:
        raise ValueError(f"Unknown site '{site}'.  Must be one of R, A, M or TC.")
    return f"fnlp/Llama3_1-8B-Base-LX{site}-{expansion}x"


def download_sae(
    layer: int,
    site: str,
    expansion: int,
    cache_dir: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> Tuple[str, SaeHyperParams]:
    """Download the weights and hyperparameters for a specific Llama‑Scope SAE.

    Parameters
    ----------
    layer : int
        Zero‑based layer index of the base model.  Must be between 0 and 31
        inclusive for Llama 3.1 8B.
    site : str
        One of ``R``, ``A``, ``M`` or ``TC``.
    expansion : int
        Expansion factor (8 or 32).
    cache_dir : str, optional
        Directory in which to cache the downloaded files.  Defaults to the
        Hugging Face cache.
    repo_id : str, optional
        Override the default repository ID (resolved from ``site`` and
        ``expansion``) if you have mirrored checkpoints under a different
        namespace.

    Returns
    -------
    weights_path : str
        Path to the downloaded ``final.safetensors`` file.
    hyperparams : SaeHyperParams
        Parsed hyperparameters for the SAE.
    """
    repo_id = repo_id or resolve_sae_repo(site, expansion)
    # Build the subfolder name.  Llama‑Scope uses the pattern
    # ``Llama3_1-8B-Base-L{layer}{site}-{expansion}x``.
    subfolder = f"Llama3_1-8B-Base-L{layer}{site}-{expansion}x"
    # Download hyperparams.json
    hp_path = hf_hub_download(
        repo_id=repo_id,
        filename="hyperparams.json",
        subfolder=subfolder,
        cache_dir=cache_dir,
        resume_download=True,
    )
    hyperparams = parse_hyperparams(hp_path)
    # Download the weights.  They live in the ``checkpoints`` subdirectory and
    # are always named ``final.safetensors``.
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="final.safetensors",
        subfolder=os.path.join(subfolder, "checkpoints"),
        cache_dir=cache_dir,
        resume_download=True,
    )
    return weights_path, hyperparams


def demo(layer: int = 0, site: str = "R", expansion: int = 8, text: str = "Hello world!") -> None:
    """Download an SAE, run it on a sample sentence and print diagnostics.

    This function loads the Llama 3.1 8B base model via the Hugging Face
    ``transformers`` library, extracts hidden states from the specified layer
    and site, applies the corresponding SAE and prints the largest sparse
    activations.  Because Llama 3.1 8B is large, this demonstration will run
    slowly and may require a GPU with >20 GB of VRAM.  For small‑scale testing,
    swap in a smaller model such as Llama 3.1 Instruct or Gemma.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve and download the SAE
    weights_path, hyperparams = download_sae(layer=layer, site=site, expansion=expansion)
    sae = TopKSparseAutoEncoder(weights_path, hyperparams)
    sae.eval()

    # Load the base language model and tokenizer
    model_name = hyperparams.device  # hyperparams only stores the training device; default to meta‑llama
    # Override with the model_name from lm_config.json if available
    try:
        lm_config_path = hf_hub_download(
            repo_id=resolve_sae_repo(site, expansion),
            filename="lm_config.json",
            subfolder=f"Llama3_1-8B-Base-L{layer}{site}-{expansion}x",
        )
        with open(lm_config_path, "r", encoding="utf-8") as f:
            lm_conf = json.load(f)
            model_name = lm_conf.get("model_name", model_name)
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Encode the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Forward pass with caching of hidden states.  We rely on the fact that
    # Transformers exposes the intermediate hidden states if `output_hidden_states=True`.
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (embedding + layer outputs)

    # Select the hidden state corresponding to the requested layer and site.
    # Llama‑Scope SAEs use the post‑MLP residual stream (hook_resid_post), so
    # we take the layer+1 hidden state.  For other sites (A, M, TC) you would
    # need to access the appropriate tensor via hooks.  Here we simplify by
    # using hidden_states[layer+1] for demonstration purposes.
    if site == "R":
        h = hidden_states[layer + 1].squeeze(0)  # (seq, d_model)
    else:
        raise NotImplementedError(
            "This demo only extracts the residual stream; to use A/M/TC sites you must register hooks on the model."
        )

    # Flatten sequence dimension
    h_flat = h.reshape(-1, h.size(-1))
    # Apply SAE
    recon, sparse, indices, activations = sae(h_flat)
    # Report the most active features for the first token
    top_indices = indices[0][:10].tolist()
    top_vals = sparse[0][top_indices].tolist()
    print(f"Top features for first token of '{text}':")
    for idx, val in zip(top_indices, top_vals):
        print(f"  feature {idx}: activation {val:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Llama‑Scope SAE on sample text.")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (0‑31).")
    parser.add_argument("--site", type=str, default="R", choices=["R", "A", "M", "TC"], help="SAE site.")
    parser.add_argument("--expansion", type=int, default=8, choices=[8, 32], help="Expansion factor (8 or 32).")
    parser.add_argument("--text", type=str, default="Hello world!", help="Text to process.")
    args = parser.parse_args()
    demo(layer=args.layer, site=args.site, expansion=args.expansion, text=args.text)


if __name__ == "__main__":
    main()
