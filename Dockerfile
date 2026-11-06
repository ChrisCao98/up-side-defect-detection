FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch==2.5.0+cu121 torchvision==0.20.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# wide_resnet50_2 权重不打进镜像，运行时挂载：
#   -v "$PWD/weights:/app/weights:ro"
CMD ["python", "train_ext.py"]
