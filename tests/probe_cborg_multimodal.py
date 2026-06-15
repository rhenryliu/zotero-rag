"""Probe whether CBORG actually parses attached images for the configured model.

CBORG is an OpenAI-compatible LiteLLM proxy whose backend is Claude. Images are
sent as OpenAI ``image_url`` data URIs (see ``rag.generation._generate_openai``);
whether they reach the model depends on LiteLLM transcoding them to Anthropic
image blocks for the ``CBORG_MODEL`` alias. This script verifies that empirically,
independent of the RAG pipeline, via two signals:

1. Canary: render a random token into a PNG, send ONLY that image, ask the model
   to read it. Returned verbatim => vision works end to end.
2. Token accounting: compare ``usage.prompt_tokens`` for the same text prompt with
   vs. without the image. A jump confirms the image was tokenized.

Usage:
    conda activate zotero-rag
    python tests/probe_cborg_multimodal.py
"""

from __future__ import annotations

import base64
import io
import os
import secrets

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openai

from rag.config import CBORG_BASE_URL, CBORG_MODEL, GEN_TEMPERATURE


def make_canary_png(token: str) -> bytes:
    """Render ``token`` as large centered text on a white PNG, returned as bytes."""
    fig = plt.figure(figsize=(6, 2), dpi=100)
    fig.text(0.5, 0.5, token, ha="center", va="center", fontsize=48, family="monospace")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def chat(client, content, *, max_tokens: int = 64):
    """Issue one chat-completions call; return ``(text, prompt_tokens)``."""
    resp = client.chat.completions.create(
        model=CBORG_MODEL,
        max_tokens=max_tokens,
        temperature=GEN_TEMPERATURE,
        messages=[{"role": "user", "content": content}],
    )
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    return (resp.choices[0].message.content or "").strip(), prompt_tokens


def main() -> None:
    token = os.environ.get("CBORG_API_KEY")
    if not token:
        raise SystemExit("$CBORG_API_KEY is not set in the environment.")
    client = openai.OpenAI(api_key=token, base_url=CBORG_BASE_URL)

    canary = secrets.token_hex(4).upper()  # e.g. "A1B2C3D4" -- not guessable
    png = make_canary_png(canary)
    data_uri = f"data:image/png;base64,{base64.standard_b64encode(png).decode('ascii')}"
    question = (
        "There is an image attached. Reply with ONLY the exact text shown in the "
        "image, no other words. If you cannot see any image, reply exactly: NO IMAGE."
    )

    print(f"Model        : {CBORG_MODEL}")
    print(f"Canary token : {canary}\n")

    with_img, tok_with = chat(
        client,
        [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": question},
        ],
    )
    without_img, tok_without = chat(client, question)

    print(f"With image     -> {with_img!r}  (prompt_tokens={tok_with})")
    print(f"Without image  -> {without_img!r}  (prompt_tokens={tok_without})\n")

    saw_canary = canary in with_img.upper()
    token_jump = (
        tok_with is not None and tok_without is not None and tok_with > tok_without + 50
    )
    print(f"Canary read back : {'YES' if saw_canary else 'no'}")
    print(f"Token count jumped: {'YES' if token_jump else 'no'} "
          f"(+{tok_with - tok_without} tokens)" if token_jump and tok_with and tok_without
          else f"Token count jumped: {'YES' if token_jump else 'no'}")
    print()
    if saw_canary:
        print("VERDICT: CBORG IS parsing attached images for this model.")
    elif token_jump:
        print("VERDICT: image was tokenized but not read back -- inconclusive; "
              "inspect the responses above.")
    else:
        print("VERDICT: images appear to be DROPPED by CBORG for this model.")


if __name__ == "__main__":
    main()
