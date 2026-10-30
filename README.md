# GroundedQA

**Answers only from your text — extractive + generative QA.**

Give GroundedQA a paragraph and a question. The answer comes **only** from
your paragraph — never from the model's outside knowledge. It offers two
answer modes:

| Mode | How it works | Response |
| --- | --- | --- |
| **Extractive** (DistilBERT) | A DistilBERT model fine-tuned on SQuAD finds the exact answer span inside your paragraph. | Answer text + confidence score + character offsets, highlighted in the paragraph. |
| **Generative AI** (Gemini) | Google Gemini writes the answer in its own words, strictly grounded in your paragraph (see the grounding rule below). | Answer text + a `grounded` flag. |

## The grounding rule

In Generative AI mode the model is instructed to use **only** the provided
paragraph. If the answer is not present in — and cannot be derived from —
the paragraph, it must refuse with exactly this message instead of answering
from outside knowledge:

> "I can't answer that from the provided content — the answer is not in your paragraph."

In that case the API returns `"grounded": false`. The full system
instruction lives in `backend/genai.py` (the `SYSTEM_INSTRUCTION` constant),
so it is easy to read and edit.

Example: paste a paragraph about NIT Patna and ask *"What is the capital of
India?"* — Generative AI mode will refuse, even though the model "knows"
the answer, because it is not in your paragraph.

## How it works

```
                 ┌──────────────────────────────────────────────┐
                 │              FastAPI backend                 │
  Frontend       │                                              │
  (index.html)   │   mode = "extractive"                        │
  paragraph + ───┼─▶ DistilBERT (SQuAD) → start/end logits ──▶ span + score
  question +     │                                              │
  mode           │   mode = "genai"                             │
                 └─▶ Gemini (grounded system instruction) ────▶ answer + grounded flag
```

- `POST /predict` accepts `{"context", "question", "mode"}` where `mode`
  is `"extractive"` (default) or `"genai"`.
- Extractive responses: `{"answer", "score", "start", "end", "mode": "extractive"}`
  (`start`/`end` are character offsets of the answer in the context).
- Generative responses: `{"answer", "mode": "genai", "grounded": true|false}`.

## Credit

Built on **Vishnu Kumar's** *Stanford Question Answering System* project —
DistilBERT fine-tuned on SQuAD v1.1 with PyTorch + Hugging Face Transformers:
<https://github.com/its-vishnu-nitp/stanford-question-answering-system>

This project turns that notebook into a deployable website: the notebook's
extractive inference approach (tokenize → start/end logits → extract span)
is implemented in `backend/model.py`, and a grounded generative mode
(Gemini) is added alongside it. No training code runs in the server.

## Model notes

- **Extractive:** `distilbert-base-uncased-distilled-squad` — DistilBERT
  already fine-tuned on SQuAD, from Hugging Face Hub. Inference runs on the
  **CPU build of PyTorch**. On first run the backend downloads the model and
  **saves it into `backend/model/`**; later startups load from that folder
  (faster, works offline). The folder is gitignored (~250 MB of weights).
- **Generative:** `gemini-2.5-flash` via the official `google-genai` Python
  SDK, temperature 0.2. Requires an API key (next section).

## Setting up the Generative AI mode (GEMINI_API_KEY)

The Gemini API key is read **only** from the `GEMINI_API_KEY` environment
variable — it is never hardcoded and must never be committed to git
(`.env` is gitignored).

1. Get a free key from Google AI Studio: <https://aistudio.google.com/apikey>
2. Provide it to the backend in ONE of these ways:
   - **Local:** copy `backend/.env.example` to `backend/.env` and set
     `GEMINI_API_KEY=your-key`, **or** export it in your shell:
     `export GEMINI_API_KEY=your-key`
   - **Hugging Face Spaces:** Space → *Settings → Variables and secrets* →
     add a **secret** named `GEMINI_API_KEY`.
3. Restart the backend. Check `GET /health` — it reports
   `"genai_configured": true`.

Without a key, Generative AI mode returns a clean error
(`HTTP 503`, "not configured") and **Extractive mode keeps working**.

## Run locally

Prerequisites: Python 3.10+ and an internet connection for the first run
(model download).

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install the CPU build of PyTorch first (much smaller than the CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install the remaining dependencies
pip install -r backend/requirements.txt

# 4. (Optional, for Generative mode) set your Gemini key
cp backend/.env.example backend/.env   # then edit backend/.env

# 5. Start the backend (from the backend folder)
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

The first startup downloads and caches the extractive model — give it a
minute. Then:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"genai_configured":false}
```

Open `frontend/index.html` in a browser (or serve it:
`cd frontend && python3 -m http.server 5500`).

Try the API directly:

```bash
# Extractive mode (default)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"context": "The National Institute of Technology Patna is a public technical university located in Patna, Bihar, India.", "question": "Where is the National Institute of Technology Patna located?"}'
# {"answer":"Patna, Bihar, India","score":0.9...,"start":...,"end":...,"mode":"extractive"}

# Generative mode (needs GEMINI_API_KEY; refuses off-topic questions)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"context": "The National Institute of Technology Patna is a public technical university located in Patna, Bihar, India.", "question": "What is the capital of India?", "mode": "genai"}'
# {"answer":"I can't answer that from the provided content — the answer is not in your paragraph.","mode":"genai","grounded":false}
```

## Deployment

### Backend → Hugging Face Spaces

1. Create a free account at <https://huggingface.co> and create a new
   **Space** with SDK = **Docker** (Blank template).
2. Put the contents of the `backend/` folder (`app.py`, `model.py`,
   `genai.py`, `requirements.txt`, `Dockerfile`, `.env.example`) into the
   Space repository and push. The Dockerfile installs the CPU PyTorch wheel
   and starts the API with `uvicorn app:app --host 0.0.0.0 --port 7860`.
3. The model folder is **not** uploaded — on the Space's first startup the
   model is downloaded from Hugging Face Hub automatically.
4. (Optional) Add the `GEMINI_API_KEY` secret in the Space settings to
   enable Generative AI mode.
5. Your backend URL will look like:
   `https://<username>-groundedqa.hf.space`

### Frontend → Vercel

1. Import this repository into Vercel and set the **Root Directory** to
   `frontend` (plain static site — no build step).
2. Before deploying, edit `frontend/script.js` and set `API_BASE_URL` to
   your Hugging Face Spaces backend URL (see the comment at the top of
   that file).

## Project structure

```
extractive-question-answering/          (repo name: groundedqa)
├── backend/
│   ├── app.py            # FastAPI app: /, /health, /predict (mode-aware, CORS on)
│   ├── model.py          # extractive model download/cache + span extraction
│   ├── genai.py          # Gemini generative mode + grounding system instruction
│   ├── requirements.txt
│   ├── Dockerfile        # for Hugging Face Spaces (port 7860)
│   ├── .env.example      # copy to .env and set GEMINI_API_KEY (never commit .env)
│   └── model/            # cached DistilBERT weights (first run, gitignored)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js         # mode selector + API calls; set API_BASE_URL after deploy
└── README.md
```
