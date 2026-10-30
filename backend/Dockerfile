# Backend image for Hugging Face Spaces (Docker SDK, port 7860).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface

WORKDIR /app

COPY requirements.txt .

# Install the CPU-only PyTorch wheel first (much smaller than the CUDA build),
# then the remaining dependencies.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app.py model.py genai.py ./

# NOTE: the model/ folder is intentionally NOT part of this image.
# On first startup model.py downloads the model from Hugging Face Hub
# and caches it locally inside the container.
#
# For the Generative AI mode, set GEMINI_API_KEY as a Space secret
# (Settings -> Variables and secrets). Without it, genai mode returns
# HTTP 503 and extractive mode keeps working.

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
