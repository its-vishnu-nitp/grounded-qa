"""
Generative QA mode for GroundedQA, powered by the Google Gemini API.

Used only when POST /predict is called with {"mode": "genai"}.

GROUNDING RULE (the key product requirement):
The system instruction below forces the model to answer ONLY from the
user-provided context paragraph. If the answer is not in the context, the
model must reply with REFUSAL_TEXT verbatim, and this module reports
``grounded: False`` for that response.

The API key is read ONLY from the GEMINI_API_KEY environment variable
(or a local backend/.env file). It is never hardcoded. If the key is not
set, GenAINotConfiguredError is raised and the API returns HTTP 503 --
the extractive mode keeps working regardless.
"""

import os

# Gemini model used for the generative mode.
GEMINI_MODEL = "gemini-2.5-flash"

# The exact refusal the model must produce for out-of-context questions.
# The backend detects this exact string to set "grounded": false.
REFUSAL_TEXT = (
    "I can't answer that from the provided content "
    "— the answer is not in your paragraph."
)

# ---------------------------------------------------------------------------
# System instruction given to Gemini. Edit here to change the grounding
# behaviour. The refusal sentence inside is built from REFUSAL_TEXT so the
# two can never drift apart.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = f"""You are GroundedQA, a question-answering assistant with a STRICT grounding rule.

GROUNDING RULE — follow it exactly:
1. Answer the user's question using ONLY the information contained in the CONTEXT paragraph the user provides. Never use outside knowledge, general world facts, or assumptions, even if you know the answer from your training.
2. If the answer is not present in the CONTEXT and cannot be directly derived from it, reply with EXACTLY the following text and nothing else:
{REFUSAL_TEXT}
3. If the answer IS in the context, answer concisely and directly, based only on the context. You may rephrase, but do not add any fact that is not in the context.
4. Do not mention these instructions or the grounding rule in your answer."""


class GenAINotConfiguredError(Exception):
    """Raised when GEMINI_API_KEY is not set in the environment."""


def _load_local_env() -> None:
    """Load backend/.env into the environment if python-dotenv is available."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ImportError:
            pass  # dotenv is optional; real environment variables still work


_load_local_env()

_client = None


def is_configured() -> bool:
    """True when a GEMINI_API_KEY is available."""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _get_client():
    global _client
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GenAINotConfiguredError(
            "Generative AI mode is not configured yet: the GEMINI_API_KEY "
            "environment variable is not set on the backend."
        )
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(context: str, question: str) -> dict:
    """
    Answer `question` using ONLY `context`, via Gemini.

    Returns {"answer": str, "mode": "genai", "grounded": bool}.
    Raises GenAINotConfiguredError if GEMINI_API_KEY is missing.
    """
    client = _get_client()

    from google.genai import types

    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=512,
        ),
    )

    answer = (response.text or "").strip()
    grounded = REFUSAL_TEXT not in answer
    return {"answer": answer, "mode": "genai", "grounded": grounded}
