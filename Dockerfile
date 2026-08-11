FROM docker.io/nvidia/cuda:13.3.1-cudnn-runtime-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN install -d -m 0755 /cuda-tile-project
WORKDIR /cuda-tile-project

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --requirement /cuda-tile-project/requirements.txt

COPY cuda_tile_runner.py run_matmul.py cuda-tile-translate ./
COPY solutions ./solutions
COPY tests ./tests
COPY matmul.mlir ./matmul.mlir

RUN chmod -R a+rX /cuda-tile-project \
    && chmod a+rx /cuda-tile-project/cuda-tile-translate

CMD ["/bin/bash"]
