"""Answer generation: provider fan-out over Ollama, Anthropic, and CBORG.

Three separate generation paths behind a single :func:`generate` dispatcher.
``anthropic`` uses the Anthropic SDK (Messages API). ``cborg`` (LBNL gateway) is
an OpenAI-compatible LiteLLM proxy, so it uses the OpenAI SDK
(``chat.completions``) with a bearer token and custom ``base_url`` -- it is *not*
Anthropic-API-compatible despite generating with Claude. Ollama has its own path.
All three pass ``GEN_TEMPERATURE``.

The caller (``RAGPipeline._generate``) builds the system prompt, the
sources-plus-question user prompt, and any page images; this module only routes
and issues the call.
"""

from __future__ import annotations

import base64
import os
import sys
from typing import Any

import ollama

from .config import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    CBORG_BASE_URL,
    CBORG_MAX_TOKENS,
    CBORG_MODEL,
    GEN_MODEL,
    GEN_NUM_CTX,
    GEN_PROVIDER,
    GEN_TEMPERATURE,
    MAX_HISTORY_MESSAGES,
)


def generate(system: str, prompt: str, images: list[bytes], history) -> str:
    """Generate a grounded answer via the configured provider.

    Args:
        system: System prompt.
        prompt: User prompt text (sources followed by the question).
        images: Page images as raw PNG bytes, or empty for text-only.
        history: Prior chat messages (role/content dicts), or None.

    Returns:
        The answer text (without the appended source list).
    """
    if GEN_PROVIDER == "anthropic":
        return _generate_anthropic(system, prompt, images, history)
    elif GEN_PROVIDER == "cborg":
        return _generate_cborg(system, prompt, images, history)
    elif GEN_PROVIDER == "ollama":
        return _generate_ollama(system, prompt, images, history)
    else:
        raise ValueError(f"Unsupported GEN_PROVIDER: {GEN_PROVIDER}")


def _generate_ollama(system: str, prompt: str, images: list[bytes], history) -> str:
    """Generate with a local Ollama chat model.

    Images are attached to the user message as raw PNG bytes (Ollama's native
    multimodal format), with a note appended to the prompt so the model knows
    to read them.

    Args:
        system: System prompt.
        prompt: User prompt text (sources followed by the question).
        images: Page images as raw PNG bytes, or empty for text-only.
        history: Prior chat messages (role/content dicts), or None.

    Returns:
        The model's reply text, stripped.
    """
    user_message: dict = {"role": "user", "content": prompt}
    if images:
        user_message["images"] = images
        user_message["content"] = (
            prompt + "\n\n(Page images of these sources are attached; use their "
            "figures and tables as needed.)"
        )
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append(user_message)
    resp = ollama.chat(
        model=GEN_MODEL,
        messages=messages,
        think=False,
        options={"temperature": GEN_TEMPERATURE, "num_ctx": GEN_NUM_CTX},
    )
    content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
    return (content or "").strip()


def _generate_anthropic(system: str, prompt: str, images: list[bytes], history) -> str:
    """Generate with the Claude API via the Anthropic Messages API.

    Builds an ``anthropic.Anthropic`` client (which reads ``ANTHROPIC_API_KEY``
    from the environment) and issues a single Messages call against
    ``ANTHROPIC_MODEL``. Images are attached as base64 ``image`` content blocks
    before the prompt text.

    Args:
        system: System prompt (passed as the top-level ``system`` parameter).
        prompt: User prompt text, appended after any image blocks.
        images: Page images as raw PNG bytes, or empty for text-only.
        history: Prior chat messages (role/content dicts), or None.

    Returns:
        The concatenated text of the response's text blocks, stripped.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
    blocks: list[dict] = []
    for img in images:
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(img).decode("ascii"),
                },
            }
        )
    blocks.append({"type": "text", "text": prompt})
    user_message: dict = {"role": "user", "content": blocks}
    messages: list[Any] = []
    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append(user_message)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        system=system,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _generate_cborg(system: str, prompt: str, images: list[bytes], history) -> str:
    """Generate via CBORG (LBNL), an OpenAI-compatible LiteLLM gateway.

    CBORG exposes the OpenAI chat-completions API (not the Anthropic Messages
    API), authenticated with a bearer token (``$CBORG_API_KEY``) against
    ``CBORG_BASE_URL``. The client is built with explicit ``api_key`` and
    ``base_url`` so it does not depend on whatever ``OPENAI_*`` variables
    happen to be exported in the shell, then delegated to
    :func:`_generate_openai`.

    Args:
        system: System prompt.
        prompt: User prompt text (sources followed by the question).
        images: Page images as raw PNG bytes, or empty for text-only.
        history: Prior chat messages (role/content dicts), or None.

    Returns:
        The generated answer text, stripped.

    Raises:
        RuntimeError: If ``$CBORG_API_KEY`` is not set in the environment.
    """
    import openai

    token = os.environ.get("CBORG_API_KEY")
    if not token:
        raise RuntimeError(
            "GEN_PROVIDER='cborg' but $CBORG_API_KEY is not set in the environment."
        )
    if images:
        # CBORG (LiteLLM) transcodes OpenAI image_url parts into Anthropic image
        # blocks, verified to reach vision models like anthropic/claude-sonnet
        # (see tests/probe_cborg_multimodal.py). A text-only CBORG_MODEL would
        # silently drop them, so flag that the images only land if the model is
        # vision-capable rather than claiming blanket failure.
        print(
            f"NOTE: MULTIMODAL is on with GEN_PROVIDER='cborg'; attached page "
            f"images are only used if CBORG_MODEL ('{CBORG_MODEL}') is "
            f"vision-capable.",
            file=sys.stderr,
        )
    client = openai.OpenAI(api_key=token, base_url=CBORG_BASE_URL)
    return _generate_openai(
        client, CBORG_MODEL, CBORG_MAX_TOKENS, system, prompt, images, history
    )


def _generate_openai(
    client,
    model: str,
    max_tokens: int,
    system: str,
    prompt: str,
    images: list[bytes],
    history,
) -> str:
    """Run one OpenAI chat-completions call (shared by OpenAI-compatible providers).

    Generic worker for any OpenAI-compatible endpoint: the caller supplies a
    configured client, model id, and token budget (e.g. :func:`_generate_cborg`
    points it at the CBORG gateway). Images are attached as base64 data-URI
    ``image_url`` parts before the prompt text.

    Args:
        client: A configured ``openai.OpenAI`` client (api key and base URL
            already set by the caller).
        model: Model identifier or gateway alias to generate with.
        max_tokens: Maximum number of tokens to generate.
        system: System prompt, sent as a leading ``system`` message.
        prompt: User prompt text, appended after any image parts.
        images: Page images as raw PNG bytes, or empty for text-only.
        history: Prior chat messages (role/content dicts), or None.

    Returns:
        The response message content, stripped.
    """
    if images:
        # Images before the prompt text (matches _generate_anthropic and the
        # convention that a vision model attends to images, then the question).
        content: list[dict] = []
        for img in images:
            data = base64.standard_b64encode(img).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{data}"},
                }
            )
        content.append({"type": "text", "text": prompt})
        user_message: dict = {"role": "user", "content": content}
    else:
        user_message = {"role": "user", "content": prompt}
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append(user_message)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=GEN_TEMPERATURE,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()
