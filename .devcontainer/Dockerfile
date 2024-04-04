# Adjusting the base image for compatibility with the requested PyTorch and CUDA versions
ARG PYTORCH="1.11.0" # Updated to required PyTorch version
ARG CUDA="11.3" # Assuming compatibility with PyTorch 1.11.0, adjust as necessary
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# The remaining Dockerfile setup stays the same until the RUN pip install ... command

# Install the required Python packages with the specified versions
RUN pip install openmim && \
    mim install mmcv-full==1.6.0 mmdet==2.25.2 mmrotate==0.3.2 && \
    pip install torch==1.11.0 torchvision==0.12.0

# The COPY, WORKDIR, and subsequent instructions remain unchanged.


RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the required packages including vim
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# \&& pip install --no-cache-dir -e .

WORKDIR /mmdetection-2.x

