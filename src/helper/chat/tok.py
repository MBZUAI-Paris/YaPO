"""
Utilities for serialising conversations to token IDs.
Adapted from https://github.com/nrimsky/CAA/blob/main/utils/tokenize.py
"""
from typing import List, Optional, Tuple

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def tokenize(
    tokenizer,
    system_prompt: str,
    model_name: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos: bool = False,
) -> List[int]:
    """
    Serialize a conversation with optional system prompt into token IDs.

    tokenizer:
        Hugging Face tokenizer used for encoding.
    system_prompt:
        Text prepended to the first user turn when supported.
    model_name:
        Used to determine how to embed the system prompt (llama-2 vs. mistral).
    conversation:
        Sequence of (user_input, model_output) tuples. `model_output` may be
        ``None`` for the final turn when preparing inputs for generation.
    no_final_eos:
        When ``True``, omit the closing EOS token on the final assistant turn.
    """

    def _instruction_response_to_tokens(
        instruction: str,
        model_output: Optional[str] = None,
        is_first_message: bool = False,
        no_eos: bool = False,
    ) -> List[int]:
        if is_first_message:
            if model_name == "llama-2":
                dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
            elif model_name == "mistral":
                dialog_content = f"{system_prompt}\n{instruction.strip()}"
            else:
                raise SystemExit("Unsupported model name: ", model_name)
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )
            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens: List[int] = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens
