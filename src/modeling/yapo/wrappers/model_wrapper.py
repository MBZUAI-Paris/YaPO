"""
This file includes code adapted from https://github.com/nrimsky/CAA/blob/main/llama_wrapper.py
Updated to:
- Accept any HF model name or local path
- Use HF chat templates via tokenizer.apply_chat_template (no tok.py dependency)
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from helper.chat import ensure_tokenizer_has_chat_template

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None

        self.save_internal_decodings = False
        self.do_projection = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (t.norm(
                last_token_activations
            )  * t.norm(self.calc_dot_product_with))
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            output = (output[0]  +  self.add_activations,) + output[1:]
        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, do_projection=False):
        self.add_activations = activations
        self.do_projection = do_projection

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.do_projection = False
        self.calc_dot_product_with = None
        self.dot_products = []


def _supports_system_role(tokenizer) -> bool:
    try:
        test_messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hello"},
        ]
        tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
        return True
    except Exception as e:
        return False


def _conversation_to_messages(system_prompt: str, history: List[Tuple[str, Optional[str]]], tokenizer) -> List[dict]:
    messages: List[dict] = []
    sys_supported = _supports_system_role(tokenizer)
    if system_prompt and sys_supported:
        messages.append({"role": "system", "content": system_prompt})

    for i, (user_text, assistant_text) in enumerate(history):
        if i == 0 and system_prompt and not sys_supported:
            # fold system into the first user turn if system not supported
            user_text = f"{system_prompt}\n\n{user_text}".strip()
            # user_text = f"{user_text}".strip()
        messages.append({"role": "user", "content": user_text})
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


class ModelWrapper:
    def __init__(
        self,
        token,
        system_prompt,
        model_name_or_path,
        generation_params: Optional[dict] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        # Backward-compat: map known aliases else accept any path/repo id
        alias_map = {
            'llama-2': 'meta-llama/Llama-2-7b-chat-hf',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
        }
        self.model_name_path = alias_map.get(model_name_or_path, model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_path, use_auth_token=token)
        ensure_tokenizer_has_chat_template(self.tokenizer, self.model_name_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_path, use_auth_token=token)
        self.model = self.model.to(self.device)
        # Some chat templates use special tokens; END_STR retained for legacy compatibility
        try:
            self.END_STR = t.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)
        except Exception:
            self.END_STR = None
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

        # Default generation parameters (can be overridden via set_generation_params)
        default_gen = {
            "do_sample": True,
            # "temperature": 0.7,
            # "top_p": 0.9,
            # "top_k": 50,
            # "repetition_penalty": 1.1,
            # optionally: "no_repeat_ngram_size": 0 (disabled)
        }
        self.generation_kwargs = {**default_gen, **(generation_params or {})}
        # Ensure EOS/PAD are set to avoid warnings for some tokenizers
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            self.generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            # fallback pad to eos if pad is missing
            self.generation_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def _build_inputs(self, history: Tuple[str, Optional[str]], add_generation_prompt: bool = True):
        messages = _conversation_to_messages(self.system_prompt, history, self.tokenizer)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        return inputs

    def _build_inputs_batch(self, histories: List[List[Tuple[str, Optional[str]]]], add_generation_prompt: bool = True):
        texts = []
        for hist in histories:
            messages = _conversation_to_messages(self.system_prompt, hist, self.tokenizer)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            texts.append(text)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        return inputs

    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        inputs = self._build_inputs([(prompt, None)], add_generation_prompt=True)
        return self.generate(inputs, max_new_tokens=max_new_tokens)

    
    def generate_text_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens=50
    ) -> str:
        inputs = self._build_inputs(history, add_generation_prompt=True)
        return self.generate(inputs, max_new_tokens=max_new_tokens)

    def generate_text_do_sample_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens=50
    ) -> str:
        inputs = self._build_inputs(history, add_generation_prompt=True)
        return self.generate_do_sample(inputs, max_new_tokens=max_new_tokens)
    
    def _prepare_generation_kwargs(self, max_new_tokens: int, override: Optional[dict] = None):
        # Merge defaults with overrides and remove sampling-only args when do_sample is False
        cfg = {**self.generation_kwargs, **(override or {})}
        cfg["max_new_tokens"] = max_new_tokens
        if not cfg.get("do_sample", False):
            # Avoid warnings from HF when do_sample is False
            for k in ["top_k", "top_p", "temperature", "typical_p"]:
                cfg.pop(k, None)
        return cfg

    def set_generation_params(self, **kwargs):
        """Update default generation parameters (e.g., repetition_penalty=1.2)."""
        self.generation_kwargs.update(kwargs)

    def generate(self, inputs, max_new_tokens=50):
        with t.no_grad():
            gen_kwargs = self._prepare_generation_kwargs(max_new_tokens)
            generated = self.model.generate(**inputs, **gen_kwargs)
            # Decode only the newly generated tokens (strip the prompt)
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
            return decoded

    def generate_do_sample(self, inputs, max_new_tokens=50):
        with t.no_grad():
            gen_kwargs = self._prepare_generation_kwargs(max_new_tokens, {"do_sample": True})
            generated = self.model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
            return decoded

    def generate_text_with_conversation_history_batch(
        self, histories: List[List[Tuple[str, Optional[str]]]], max_new_tokens: int = 50
    ) -> List[str]:
        inputs = self._build_inputs_batch(histories, add_generation_prompt=True)
        with t.no_grad():
            gen_kwargs = self._prepare_generation_kwargs(max_new_tokens)
            generated = self.model.generate(**inputs, **gen_kwargs)
        # Decode per-example new tokens only
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs: List[str] = []
        for i, L in enumerate(input_lens):
            txt = self.tokenizer.decode(generated[i, L:], skip_special_tokens=True)
            outputs.append(txt)
        return outputs

    def generate_text_batch(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        histories = [[(p, None)] for p in prompts]
        return self.generate_text_with_conversation_history_batch(histories, max_new_tokens=max_new_tokens)

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_logits_with_conversation_history(self, history: Tuple[str, Optional[str]]):
        inputs = self._build_inputs(history, add_generation_prompt=False)
        return self.get_logits(inputs["input_ids"])

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, do_projection=False):
        self.model.model.layers[layer].add(activations, do_projection)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )


    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
