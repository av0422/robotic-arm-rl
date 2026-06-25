# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies for MuJoCo rendering (EGL / osmesa headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

# Pre-create output dirs
RUN mkdir -p videos models

# ── Headless MuJoCo renderer ──────────────────────────────────────────────────
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Default: launch the Gradio demo
EXPOSE 7860
CMD ["python", "app.py"]
