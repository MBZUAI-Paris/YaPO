"""
SparseModelWrapper: a sparse-steering-capable wrapper mirroring the dense ModelWrapper API.

It injects a sparse steering vector (in SAE latent space) at a specified transformer
layer, using an SAE encoder/decoder to map to/from the model's hidden space.

This is designed to be API-compatible with code that expects:
 - .generate_text / .generate_text_with_conversation_history
 - .set_add_activations(layer, activations, ...)
 - .reset_all()

The actual sparse injection is implemented in SparseSteerBlockWrapper.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from helper.chat import ensure_tokenizer_has_chat_template
from sae import load_sae


SYSTEM_PROMPT = ""


class SparseSteerBlockWrapper(t.nn.Module):
    """Wraps a transformer block and applies sparse steering post-block using an SAE.

    The injected vector is expected to be in the SAE's sparse space and is added to
    the encoded hidden activations, then decoded back to hidden space. A delta term
    preserves reconstruction fidelity similar to the training-time approach:

        final_hidden = decode(encode(hidden) + steer_vec) + (hidden - decode(encode(hidden)))
    """

    def __init__(self, block: t.nn.Module, sae: t.nn.Module, sparse_dim: int):
        super().__init__()
        self.block = block
        self.sae = sae
        self.sparse_dim = sparse_dim

        # Injection state: vector in sparse space (1D) or None
        self.add_activations: Optional[t.Tensor] = None

        # SAE should be eval + frozen for inference
        self.sae.eval()
        for p in self.sae.parameters():
            p.requires_grad = False

    def _encode_sparse(self, hidden_flat: t.Tensor) -> t.Tensor:
        encoded = self.sae.encode(hidden_flat)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        return encoded

    def forward(self, *args, **kwargs):
        # Run original block
        output = self.block(*args, **kwargs)
        hidden = output[0]  # [B, T, H]

        if self.add_activations is None:
            return output

        B, T, H = hidden.shape
        device = hidden.device

        # Flatten to [B*T, H] for SAE
        hidden_flat = hidden.reshape(B * T, H)

        # Encode to sparse space, add steering, apply ReLU, decode back
        with t.no_grad():
            sparse = self._encode_sparse(hidden_flat)  # [B*T, S]

            steer_vec = self.add_activations
            if steer_vec.dim() == 1:
                steer_vec = steer_vec.view(1, -1)
            # Broadcast addition over tokens
            steered_sparse = sparse + steer_vec.to(sparse.dtype).to(device)
            steered_sparse = t.relu(steered_sparse)

            steered_hidden_flat = self.sae.decode(steered_sparse)

            # Reconstruction delta to preserve info outside SAE manifold
            recon_flat = self.sae.decode(sparse)
            delta = hidden_flat - recon_flat
            final_hidden_flat = steered_hidden_flat + delta

        final_hidden = final_hidden_flat.view(B, T, H)
        return (final_hidden,) + output[1:]

    def add(self, activations: Optional[t.Tensor]):
        # Expect a 1D vector in sparse space (size == self.sparse_dim) or None to clear
        self.add_activations = activations

    def reset(self):
        self.add_activations = None


def _supports_system_role(tokenizer) -> bool:
    try:
        test_messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hello"},
        ]
        tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
        return True
    except Exception:
        return False


def _conversation_to_messages(system_prompt: str, history, tokenizer):
    messages = []
    sys_supported = _supports_system_role(tokenizer)
    if system_prompt and sys_supported:
        messages.append({"role": "system", "content": system_prompt})
    for i, (user_text, assistant_text) in enumerate(history):
        if i == 0 and system_prompt and not sys_supported:
            user_text = f"{system_prompt}\n\n{user_text}".strip()
            # user_text = f"{user_text}".strip()
        messages.append({"role": "user", "content": user_text})
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


class SparseModelWrapper:
    def __init__(
        self,
        token: Optional[str],
        system_prompt: str,
        model_name_or_path: str,
        injection_layer: int,
        sae_repo: str = "google/gemma-scope-2b-pt-res",
        sae_width: str = "65k",
        sae_avg_idx: str = "68",
        sae_source: Optional[str] = None,
        llama_sae_site: str = "R",
        llama_sae_expansion: int = 8,
        llama_sae_cache_dir: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        # Accept any HF model repo or local path, plus aliases
        alias_map = {
            'llama-2': 'meta-llama/Llama-2-7b-chat-hf',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
        }
        self.model_name_path = alias_map.get(model_name_or_path, model_name_or_path)
        self.injection_layer = int(injection_layer)

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path, use_auth_token=token)
        ensure_tokenizer_has_chat_template(self.tokenizer, self.model_name_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_path, use_auth_token=token)
        self.model = self.model.to(self.device)

        # Prepare end-of-instruction marker for legacy decode paths
        self.END_STR = t.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)

        # Load SAE for the target layer
        self.sae = load_sae(
            model_path=sae_repo,
            layer_index=self.injection_layer,
            sae_vector_size=sae_width,
            avg_idx=sae_avg_idx,
            sae_source=sae_source,
            llama_site=llama_sae_site,
            llama_expansion=llama_sae_expansion,
            llama_cache_dir=llama_sae_cache_dir,
        )

        # Determine sparse dimension from SAE weights
        sparse_dim = getattr(self.sae.W_enc, "shape", [None, None])[1]
        if sparse_dim is None:
            # Fallback: try parameter size inference
            sparse_dim = self.sae.W_enc.size(1)  # type: ignore[attr-defined]

        # Wrap ONLY the injection layer with sparse steering block
        layer_module = self.model.model.layers[self.injection_layer]
        self.model.model.layers[self.injection_layer] = SparseSteerBlockWrapper(layer_module, self.sae, sparse_dim)

        # Default generation parameters
        default_gen: Dict = {
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        self.generation_kwargs: Dict = default_gen
        # Ensure EOS/PAD present when available to avoid warnings
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            self.generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.generation_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

    # --- API parity helpers ---
    def set_save_internal_decodings(self, value: bool):
        # No-op for sparse wrapper (not capturing internals by default)
        return None

    def _build_inputs(self, history, add_generation_prompt: bool = True):
        messages = _conversation_to_messages(self.system_prompt, history, self.tokenizer)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return inputs

    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self._build_inputs([(prompt, None)], add_generation_prompt=True)
        return self.generate(inputs, max_new_tokens=max_new_tokens)

    def generate_text_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens: int = 50
    ) -> str:
        inputs = self._build_inputs(history, add_generation_prompt=True)
        return self.generate(inputs, max_new_tokens=max_new_tokens)

    def generate_text_do_sample_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens: int = 50
    ) -> str:
        inputs = self._build_inputs(history, add_generation_prompt=True)
        return self.generate_do_sample(inputs, max_new_tokens=max_new_tokens)

    def _prepare_generation_kwargs(self, max_new_tokens: int, override: Optional[Dict] = None) -> Dict:
        cfg = {**self.generation_kwargs, **(override or {})}
        cfg["max_new_tokens"] = max_new_tokens
        if not cfg.get("do_sample", False):
            for k in ["top_k", "top_p", "temperature", "typical_p"]:
                cfg.pop(k, None)
        return cfg

    def set_generation_params(self, **kwargs):
        self.generation_kwargs.update(kwargs)

    def generate(self, inputs, max_new_tokens: int = 50):
        with t.no_grad():
            gen_kwargs = self._prepare_generation_kwargs(max_new_tokens)
            generated = self.model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
            return decoded

    def generate_do_sample(self, inputs, max_new_tokens: int = 50):
        with t.no_grad():
            gen_kwargs = self._prepare_generation_kwargs(max_new_tokens, {"do_sample": True})
            generated = self.model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
            return decoded

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_logits_with_conversation_history(self, history: Tuple[str, Optional[str]]):
        inputs = self._build_inputs(history, add_generation_prompt=False)
        return self.get_logits(inputs["input_ids"])

    def _build_inputs_batch(self, histories, add_generation_prompt: bool = True):
        texts = []
        for hist in histories:
            messages = _conversation_to_messages(self.system_prompt, hist, self.tokenizer)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            texts.append(text)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        return inputs

    def generate_text_with_conversation_history_batch(self, histories, max_new_tokens: int = 50):
        inputs = self._build_inputs_batch(histories, add_generation_prompt=True)
        with t.no_grad():
            gen_kwargs = self._prepare_generation_kwargs(max_new_tokens)
            generated = self.model.generate(**inputs, **gen_kwargs)
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for i, L in enumerate(input_lens):
            txt = self.tokenizer.decode(generated[i, L:], skip_special_tokens=True)
            outputs.append(txt)
        return outputs

    def generate_text_batch(self, prompts, max_new_tokens: int = 50):
        histories = [[(p, None)] for p in prompts]
        return self.generate_text_with_conversation_history_batch(histories, max_new_tokens=max_new_tokens)

    def set_add_activations(self, layer: int, activations: Optional[t.Tensor], do_projection: bool = False):
        if int(layer) != self.injection_layer:
            raise ValueError(f"SparseModelWrapper only supports injection at layer {self.injection_layer}, got {layer}.")
        wrapper = self.model.model.layers[self.injection_layer]
        if not isinstance(wrapper, SparseSteerBlockWrapper):
            raise RuntimeError("Injection layer is not wrapped by SparseSteerBlockWrapper.")
        wrapper.add(activations)

    def reset_all(self):
        layer = self.model.model.layers[self.injection_layer]
        if isinstance(layer, SparseSteerBlockWrapper):
            layer.reset()
