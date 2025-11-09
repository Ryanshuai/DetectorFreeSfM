FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y \
    git cmake build-essential wget \
    libboost-all-dev libeigen3-dev libsuitesparse-dev \
    libfreeimage-dev libgoogle-glog-dev libgflags-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libcgal-qt5-dev libmetis-dev libatlas-base-dev \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

WORKDIR /tmp
RUN git clone https://github.com/ceres-solver/ceres-solver.git \
    && cd ceres-solver && git checkout 2.1.0 \
    && mkdir build && cd build \
    && cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCXSPARSE=OFF \
    && make -j$(nproc) && make install \
    && cd /tmp && rm -rf ceres-solver

RUN git clone https://github.com/hxy-123/colmap.git \
    && cd colmap && mkdir build && cd build \
    && cmake .. -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF \
    && make -j$(nproc) && make install \
    && cd /tmp && rm -rf colmap

RUN conda create -n detectorfreesfm python=3.9 -y

SHELL ["conda", "run", "-n", "detectorfreesfm", "/bin/bash", "-c"]

RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install --no-cache-dir \
    pytorch-lightning==1.3.5 ray==2.7.1 pycolmap==0.3.0 \
    opencv-python yacs einops==0.3.0 kornia==0.4.1 timm \
    hydra-core omegaconf wandb rich natsort loguru \
    torchmetrics==0.6.0 joblib h5py \
    six protobuf albumentations

WORKDIR /workspace
RUN git clone https://github.com/zju3dv/DetectorFreeSfM.git && \
    cd DetectorFreeSfM && \
    git submodule update --init --recursive
	
WORKDIR /workspace/DetectorFreeSfM/third_party
RUN cd multi-view-evaluation && \
    mkdir build && cd build && \
    cmake .. && make -j8

ENV PYTHONPATH="/workspace/DetectorFreeSfM/third_party/RoIAlign.pytorch:$PYTHONPATH"

RUN pip install --no-cache-dir \
    setuptools==59.5.0 \
    opencv-python==4.5.5.64 \
    "numpy<2"

RUN colmap -h > /dev/null && echo "COLMAP OK"

RUN useradd -m -s /bin/bash ubuntu && \
    chown -R ubuntu:ubuntu /workspace /opt/conda
USER ubuntu

WORKDIR /workspace/DetectorFreeSfM
CMD ["conda", "run", "--no-capture-output", "-n", "detectorfreesfm", "/bin/bash"]
