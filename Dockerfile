# Signature System API — live deployment (Hugging Face Spaces, Docker SDK)
# Base: Python 3.12 — is par TensorFlow STABLE milta hai (Yasir ke PC wale
# Python 3.14 par TF 2.22.0rc0 aya tha; Docker me ye masla nahi hoga).
# Build context = yehi signature-api folder (Space repo ya GitHub repo).
# Model files (signature_model.pt + siamese.weights.h5) isi folder me hon
# to image me aa jati hain — .dockerignore me unhein roka NAHI gaya.
# .env KABHI image me nahi jati (.dockerignore) — secrets Space ke
# Settings > Secrets me dalte hain, repo me nahi.
FROM python:3.12-slim

# ultralytics / OpenCV ko system libraries chahiye hoti hain
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-verification.txt ./
# pehle requirements, phir verification (TensorFlow) — dono ek dafa install
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-verification.txt

COPY . .

# Spaces ka default port 7860 hai; $PORT mile to wahi, warna 7860
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
