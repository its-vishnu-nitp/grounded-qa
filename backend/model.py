"""
Model loading and inference for Extractive Question Answering.

Based on the inference approach of Vishnu Kumar's Stanford Question
Answering System notebook (`nlp code.ipynb`): tokenize (question, context)
with a DistilBERT model fine-tuned on SQuAD v1.1, take the start/end logits,
and extract the best answer span from the context. Training code from the
notebook is intentionally NOT included here -- this module is inference only.

The model is loaded ONCE (see load_model()) and reused for every request.
On the very first run the model is downloaded from Hugging Face Hub and
saved into the local ``model/`` folder next to this file; every later
startup loads from that folder, which is much faster and works offline.
"""

import os
import threading

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# DistilBERT already fine-tuned on SQuAD (verified on Hugging Face Hub).
MODEL_ID = "distilbert-base-uncased-distilled-squad"

# Local folder where the downloaded model is cached (gitignored).
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MAX_LENGTH = 512          # tokens (question + context), same order as the notebook
MAX_ANSWER_LENGTH = 30    # max answer length in tokens

_device = torch.device("cpu")
_tokenizer = None
_model = None
_load_lock = threading.Lock()


def _local_model_available() -> bool:
    return os.path.isdir(LOCAL_MODEL_DIR) and os.path.exists(
        os.path.join(LOCAL_MODEL_DIR, "config.json")
    )


def load_model() -> None:
    """Load tokenizer + model once. Safe to call multiple times."""
    global _tokenizer, _model
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return

        if _local_model_available():
            source = LOCAL_MODEL_DIR
            print(f"[model] Loading model from local folder: {LOCAL_MODEL_DIR}")
        else:
            source = MODEL_ID
            print(f"[model] Downloading model from Hugging Face Hub: {MODEL_ID}")

        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForQuestionAnswering.from_pretrained(source)

        if source == MODEL_ID:
            # First run: cache the model locally so future startups are fast.
            os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
            model.save_pretrained(LOCAL_MODEL_DIR)
            tokenizer.save_pretrained(LOCAL_MODEL_DIR)
            print(f"[model] Model saved to local folder: {LOCAL_MODEL_DIR}")

        model.to(_device)
        model.eval()
        _tokenizer = tokenizer
        _model = model
        print("[model] Model ready.")


def model_loaded() -> bool:
    return _model is not None


def predict(context: str, question: str) -> dict:
    """
    Extract the answer to `question` from `context`.

    Returns {"answer": str, "score": float, "start": int, "end": int}
    where start/end are character offsets of the answer inside `context`
    and score is a softmax-based confidence in [0, 1].
    """
    if _model is None or _tokenizer is None:
        load_model()

    encoding = _tokenizer(
        question,
        context,
        truncation="only_second",   # truncate the context, never the question
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding["offset_mapping"]
    sequence_ids = encoding.sequence_ids()  # 0 = question, 1 = context, None = special

    inputs = {
        key: torch.tensor([encoding[key]], dtype=torch.long)
        for key in ("input_ids", "attention_mask")
    }

    with torch.no_grad():
        outputs = _model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Only tokens belonging to the context can be part of the answer.
    context_token_idx = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    if not context_token_idx:
        return {"answer": "", "score": 0.0, "start": 0, "end": 0}

    # Pick the best (start, end) span: end >= start, length <= MAX_ANSWER_LENGTH.
    best_score = None
    best_pair = (context_token_idx[0], context_token_idx[0])
    for start_idx in context_token_idx:
        for end_idx in context_token_idx:
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > MAX_ANSWER_LENGTH:
                break
            span_score = start_logits[start_idx].item() + end_logits[end_idx].item()
            if best_score is None or span_score > best_score:
                best_score = span_score
                best_pair = (start_idx, end_idx)

    start_idx, end_idx = best_pair
    start_char = offset_mapping[start_idx][0]
    end_char = offset_mapping[end_idx][1]
    answer = context[start_char:end_char]

    # Softmax-based confidence for the chosen span.
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    score = float(start_probs[start_idx] * end_probs[end_idx])

    return {"answer": answer, "score": score, "start": int(start_char), "end": int(end_char)}
