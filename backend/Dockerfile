
# TM: Use the Python minor version validated by the notebook and API tests.
FROM python:3.11-slim

WORKDIR /app

# TM: Keep runtime output visible and the pre-downloaded model cache predictable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/huggingface \
    TOKENIZERS_PARALLELISM=false

# TM: FAISS/scikit-learn CPU wheels need the OpenMP shared library on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# TM: Install dependencies before copying application code to reuse build layers.
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip

# TM: Install the validated PyTorch CPU version without CUDA dependencies.
RUN pip install --no-cache-dir \
    torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Download the model at build time
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings( \
    model_name='HarishMaths/Hotel-Policy-Embedding', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True} \
)"

# TM: Copy only deployable code and the prepared index; never copy notebook
# TM: config files, credentials, or the entire working directory into the image.
COPY app.py rag_utils.py ./
COPY faiss_index/ ./faiss_index/

# TM: Fail the build for an empty index or missing optimized parent metadata.
# TM: Inspect pickle opcodes without executing pickle deserialization.
RUN python -c "import faiss, pickletools; from pathlib import Path; assert faiss.read_index('faiss_index/index.faiss').ntotal > 0; assert any(isinstance(argument, str) and argument == 'parent_context' for opcode, argument, position in pickletools.genops(Path('faiss_index/index.pkl').read_bytes()))"

# TM: Run the API without root privileges and allow it to read the model cache.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home appuser \
    && chown -R appuser:appuser /app
USER 10001:10001

# Force offline mode at runtime so it uses the cached model
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# TM: Health checks only contact the local API and never call a paid model.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3).close()"]

CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
