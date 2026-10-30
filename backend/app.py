"""
FastAPI backend for GroundedQA (Answers only from your text).

Two answer modes, chosen per request with the "mode" field:
  - "extractive" (default): DistilBERT fine-tuned on SQuAD extracts the
    exact answer span from the context (see model.py). Response:
    {"answer", "score", "start", "end", "mode": "extractive"}
  - "genai": Google Gemini answers using ONLY the provided context and
    refuses out-of-context questions (see genai.py). Response:
    {"answer", "mode": "genai", "grounded": bool}
    Requires the GEMINI_API_KEY environment variable; without it this
    mode returns HTTP 503 while extractive mode keeps working.

Endpoints:
  GET  /        -> service info
  GET  /health  -> health check + model/config status
  POST /predict -> {"context": str, "question": str, "mode": "extractive"|"genai"}

Run locally:
  uvicorn app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import model as qa_model
    import genai as genai_mode
except ImportError:  # pragma: no cover - when imported as part of a package
    from . import model as qa_model
    from . import genai as genai_mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the extractive model ONCE at startup (downloads + caches it
    # on first run). The Gemini client is created lazily per request.
    qa_model.load_model()
    yield


app = FastAPI(
    title="GroundedQA",
    description=(
        "Answers only from your text. Two modes: 'extractive' (DistilBERT "
        "fine-tuned on SQuAD returns the exact span) and 'genai' (Gemini, "
        "grounded strictly in your paragraph)."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# The frontend is hosted separately (Vercel), so allow all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    context: str = Field(..., description="The paragraph to answer from")
    question: str = Field(..., description="The question to answer")
    mode: Literal["extractive", "genai"] = Field(
        "extractive", description="Answer mode: 'extractive' or 'genai'"
    )


class PredictResponse(BaseModel):
    answer: str
    mode: str
    # Extractive mode only:
    score: Optional[float] = None
    start: Optional[int] = None  # char offset of the answer in the context
    end: Optional[int] = None    # char offset (exclusive) in the context
    # Generative mode only:
    grounded: Optional[bool] = None


@app.get("/")
def root():
    return {
        "name": "GroundedQA",
        "tagline": "Answers only from your text — extractive + generative QA",
        "modes": {
            "extractive": f"DistilBERT fine-tuned on SQuAD ({qa_model.MODEL_ID})",
            "genai": f"Google Gemini ({genai_mode.GEMINI_MODEL}), grounded in your paragraph only",
        },
        "genai_configured": genai_mode.is_configured(),
        "endpoints": {"health": "/health", "predict": "/predict (POST)"},
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": qa_model.model_loaded(),
        "genai_configured": genai_mode.is_configured(),
    }


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(request: PredictRequest):
    context = request.context.strip()
    question = request.question.strip()
    if not context:
        raise HTTPException(status_code=400, detail="Context must not be empty.")
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if request.mode == "genai":
        try:
            return genai_mode.generate_answer(context, question)
        except genai_mode.GenAINotConfiguredError as exc:
            # Clean 503: GenAI not set up yet; extractive mode is unaffected.
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # Gemini API / network errors
            raise HTTPException(
                status_code=502, detail=f"Generative AI request failed: {exc}"
            )

    result = qa_model.predict(context, question)
    result["mode"] = "extractive"
    return result
