"""
Modal deployment for the GroundedQA backend.

Deploys the EXISTING FastAPI app (backend/app.py, together with model.py
and genai.py) as a Modal web endpoint. Nothing about the API changes --
this file only describes the container image and the serving function.

Deploy (run from the PROJECT ROOT, i.e. ~/workspace/extractive-question-answering):

    modal deploy backend/modal_app.py

Modal prints the public URL of the endpoint, of the form:
    https://<workspace>--grounded-qa-fastapi-app.modal.run

Notes:
  * The DistilBERT QA model is BAKED INTO the image at build time (see
    _download_model below): it is downloaded once from Hugging Face Hub
    and saved to /model inside the image. At runtime MODEL_DIR=/model
    makes backend/model.py load it from disk -- no download, fast cold
    starts.
  * GEMINI_API_KEY is OPTIONAL. Deploying works without it: extractive
    mode works fully and genai mode returns a clean HTTP 503 (that is
    handled inside genai.py). To enable genai mode later:
        modal secret create gemini GEMINI_API_KEY=<your-key>
    then uncomment the SECRETS line marked below and redeploy.
    (The secret is deliberately NOT attached by default: Modal fails a
    deployment that references a secret which does not exist yet.)
"""

import os
import sys
from pathlib import Path

import modal

# This file lives in backend/. Paths are derived from __file__ (never from
# the current working directory) so `modal deploy backend/modal_app.py`
# works from the project root or anywhere else.
BACKEND_DIR = Path(__file__).resolve().parent

# Where the backend code lives inside the container image, and where the
# baked-in model lives.
REMOTE_BACKEND_DIR = "/root/backend"
MODEL_DIR_IN_IMAGE = "/model"

# Same model id as backend/model.py (MODEL_ID there).
HF_MODEL_ID = "distilbert-base-uncased-distilled-squad"


def _download_model() -> None:
    """Image-build step: download the QA model and save it into /model.

    Runs once while the Modal image is being built (network access is
    available at build time). Uses save_pretrained, exactly like
    backend/model.py does for its local cache, so the folder layout is
    identical (config.json + weights + tokenizer files).
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForQuestionAnswering.from_pretrained(HF_MODEL_ID)
    model.save_pretrained(MODEL_DIR_IN_IMAGE)
    tokenizer.save_pretrained(MODEL_DIR_IN_IMAGE)
    print(f"[modal build] Model saved to {MODEL_DIR_IN_IMAGE}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch from the CPU-only wheel index (much smaller than the CUDA build).
    .pip_install(
        "torch>=2.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    # The remaining dependencies, same set as backend/requirements.txt.
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "transformers>=4.40",
        "pydantic>=2.0",
        "google-genai>=1.0",
        "python-dotenv>=1.0",
    )
    # Bake the model into the image (layer is cached unless this changes).
    .run_function(_download_model)
    # Runtime loads the model from the baked-in folder -- see model.py.
    .env({"MODEL_DIR": MODEL_DIR_IN_IMAGE})
    # Copy the existing backend code into the image. Done LAST so editing
    # app.py / model.py / genai.py does not invalidate the model layer.
    .add_local_file(
        BACKEND_DIR / "app.py", f"{REMOTE_BACKEND_DIR}/app.py", copy=True
    )
    .add_local_file(
        BACKEND_DIR / "model.py", f"{REMOTE_BACKEND_DIR}/model.py", copy=True
    )
    .add_local_file(
        BACKEND_DIR / "genai.py", f"{REMOTE_BACKEND_DIR}/genai.py", copy=True
    )
    .workdir(REMOTE_BACKEND_DIR)
)

app = modal.App("grounded-qa")

# --- Optional Gemini secret -------------------------------------------------
# Deploying must succeed BEFORE any secret exists, so no secret is attached
# by default. After creating it with:
#     modal secret create gemini GEMINI_API_KEY=<your-key>
# uncomment the next line (and the `secrets=SECRETS` argument below), then
# redeploy with `modal deploy backend/modal_app.py`.
SECRETS = [modal.Secret.from_name("gemini")]


@app.function(
    image=image,
    cpu=2.0,
    memory=2048,
    timeout=900,
    scaledown_window=300,  # scale to zero after 5 min idle (saves credits)
    secrets=SECRETS,
)
@modal.asgi_app()
def fastapi_app():
    # The backend modules live in REMOTE_BACKEND_DIR inside the image.
    # app.py imports `model` and `genai` as top-level modules, so that
    # directory must be importable; the image workdir already points
    # there, and this insert makes it robust regardless of cwd.
    if REMOTE_BACKEND_DIR not in sys.path:
        sys.path.insert(0, REMOTE_BACKEND_DIR)
    os.chdir(REMOTE_BACKEND_DIR)

    from app import app as fastapi_application

    return fastapi_application
