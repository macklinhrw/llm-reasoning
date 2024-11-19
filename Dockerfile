# FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV SHELL=/bin/bash
ENV PYTHON_VERSION=3

# Update, upgrade, install packages and clean up
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends \
    git wget curl bash libgl1 software-properties-common openssh-server vim unzip && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3 python3-dev -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen


# Git Install
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

# Git LFS install
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs

# Set up Python and pip
RUN rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip

RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir torch torchvision torchaudio
RUN pip install --upgrade --no-cache-dir jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions

# Set up Jupyter Notebook
RUN pip install notebook==6.5.5
RUN jupyter contrib nbextension install --user && \
    jupyter nbextension enable --py widgetsnbextension

# RUN mkdir /.local && \
# chown -R 1001 /.local && \
# echo "export PATH=$PATH:~/.local/bin" > ~/.bashrc

# USER 1001

# Install python requirements and finish
RUN pip install vllm

# RUN pip install git+https://github.com/vllm-project/vllm.git@main

# Install AWS CLI for syncing
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws && \
    rm awscliv2.zip

COPY requirements.txt /workspace

RUN pip install -r ./docker-src/requirements.txt

RUN pip install -q -U transformers==4.46.0
RUN pip install -q -U flash-attn --no-build-isolation --no-deps

# RUN pip install -q -U git+https://github.com/huggingface/peft.git
# RUN pip install -q -U git+https://github.com/huggingface/trl.git
# RUN pip install -q -U git+https://github.com/huggingface/accelerate.git
# RUN pip install -q -U git+https://github.com/TimDettmers/bitsandbytes.git

COPY ./docker-src/* /workspace

RUN chmod +x ./start.sh

# CMD ["python", "./script.py"]
