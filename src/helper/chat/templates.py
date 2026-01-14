from __future__ import annotations

from textwrap import dedent
from typing import Optional

GENERIC_CHAT_TEMPLATE = dedent(
    """\
    {{ bos_token if bos_token is defined else '' }}{% for message in messages %}
    {% if message['role'] == 'system' %}
    [SYSTEM] {{ message['content'] | trim }}
    {% elif message['role'] == 'user' %}
    [USER] {{ message['content'] | trim }}
    {% elif message['role'] == 'assistant' %}
    [ASSISTANT] {{ message['content'] | trim }}
    {% endif %}{% endfor %}
    {% if add_generation_prompt %}[ASSISTANT] {% endif %}
    """
).strip()

CHAT_TEMPLATE_REGISTRY = {
    "generic": GENERIC_CHAT_TEMPLATE,
    "llama-2": GENERIC_CHAT_TEMPLATE,
    "llama-3": GENERIC_CHAT_TEMPLATE,
    "mistral": GENERIC_CHAT_TEMPLATE,
    "gemma": GENERIC_CHAT_TEMPLATE,
}


def ensure_tokenizer_has_chat_template(tokenizer, template_hint: Optional[str] = None) -> str:
    """
    Ensure ``tokenizer.chat_template`` is populated.

    If the tokenizer already defines a template, it is left untouched.
    Otherwise, a generic fallback (or template-specific override) is
    installed and returned.
    """
    existing = getattr(tokenizer, "chat_template", None)
    if existing:
        return existing

    hint = (template_hint or "generic").lower()
    template = CHAT_TEMPLATE_REGISTRY.get(hint, GENERIC_CHAT_TEMPLATE)
    tokenizer.chat_template = template
    return template
