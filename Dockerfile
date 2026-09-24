# Web demo for the flower classifier (CPU only).
#
# Build:  docker build -t flower-classifier .
# Run with a local checkpoint:
#         docker run -p 7860:7860 -v "$PWD/check_point.pt:/models/check_point.pt:ro" flower-classifier
# Run with a model published to the Hugging Face Hub:
#         docker run -p 7860:7860 -e HUB_REPO=your-name/flower-classifier flower-classifier
# Then open http://localhost:7860
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    CHECKPOINT=/models/check_point.pt

WORKDIR /app

# CPU-only PyTorch keeps the image a fraction of the size of the default CUDA build
# (versions pinned to match constraints.txt, which pins everything else)
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-demo.txt constraints.txt ./
RUN pip install -r requirements-demo.txt -c constraints.txt

COPY model_utils.py app.py cat_to_name.json ./

# Run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 7860
CMD ["python", "app.py"]
