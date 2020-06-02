FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Get BIONIC
RUN git clone https://github.com/bowang-lab/BIONIC.git
WORKDIR /BIONIC

# Create BIONIC conda environment
RUN conda env create -f environment.yml
RUN echo "source activate bionic" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin:$PATH \
    CPATH=/usr/local/cuda/include:$CPATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

# Install Pytorch Geometric
RUN /bin/bash -c "source activate bionic \
    && pip install torch-cluster==1.4.2 --no-cache-dir \
    && pip install torch-sparse==0.4.0 --no-cache-dir \
    && pip install torch-scatter==1.2.0 --no-cache-dir \
    && pip install git+https://github.com/duncster94/pytorch_geometric.git@master#egg=torch-geometric"

WORKDIR /BIONIC/src