## Introduction
BIONIC (**Bio**logical **N**etwork **I**ntegration using **C**onvolutions) is a deep-learning based biological network integration algorithm that extends the graph convolutional network (GCN) to learn integrated features for genes or proteins across input networks. BIONIC produces high-quality node features and is scalable both in number of networks and network size.

An overview of BIONIC can be seen below.

<p align="center">
  <a href="https://ibb.co/nBTSh1P"><img src="https://i.ibb.co/XD9Tm5Y/Figure-1.png" alt="BIONIC overview" border="0"></a>
</p>

1. Nodes in the input networks are given a unique, one-hot encoded feature vector.
2. Each network is passed through its own graph convolutional encoder where each node feature vector is updated based on the node's local neighbourhood.
3. These features are projected into a lower-dimensional space through a learned mapping.
4. This process yields **network-specific** node features.
5. Through a stochastically masked summation step, integrated node features are obtained.
6. Integrated features are then extracted for use in downstream tasks.
7. In order to train, BIONIC decodes the integrated features -
8. Into a reconstruction of the input networks.
9. BIONIC minimizes the difference between the reconstructed network and the input networks (i.e. reconstruction error) and by doing so, improves the quality of the integrated feature set.

BIONIC is implemented in [Python 3](https://www.python.org/downloads/) and uses [PyTorch](https://pytorch.org/).

## Installation
NOTE: Currently BIONIC requires an NVIDIA GPU to run.

### [Docker](https://www.docker.com/) (Recommended, Linux only)

If you are on a Linux machine it's recommended to run BIONIC in a Docker container. 

1. Copy or download the Dockerfile from [here](https://raw.githubusercontent.com/bowang-lab/BIONIC/master/Dockerfile) by running

        $ wget https://raw.githubusercontent.com/bowang-lab/BIONIC/master/Dockerfile

2. Build the BIONIC Docker image by running

        $ docker build -t "bionic" /path/to/Dockerfile
   
   NOTE: This may take some time.
3. Install `nvidia-container-toolkit` by running

        $ apt-get install -y nvidia-container-toolkit
        
4. Create a BIONIC instance by running

        $ docker run -it --gpus all --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "bionic" /bin/bash

5. Test BIONIC works by running the following inside the Docker container

        $ python main.py -c gav_krog.json

<!---
### TODO

1. Download and install [Anaconda](https://www.anaconda.com/distribution/) for Python 3.x.

2. Install CUDA Toolkit 10.0 from [here](https://developer.nvidia.com/cuda-10.0-download-archive). NOTE: The CUDA Toolkit version **must be 10.0**, 10.1+ will not work.
3. Locate the CUDA Toolkit installation directory (it should be something similar to `/usr/local/cuda-10.0/bin`). Add this path to the `$PATH` variable by doing
        
        $ export PATH=/usr/local/cuda-10.0/bin:$PATH

4. Add `cuda-10.0/include` to `$CPATH` by running

        $ export CPATH=/usr/local/cuda-10.0/include:$CPATH
        
5. (**Linux**) Add `cuda-10.0/lib64` to `$LD_LIBRARY_PATH` by running

        $ export LD_LIBRARY_PATH=/usr/local/cuda-10.0/bin:$LD_LIBRARY_PATH

    (**macOS**) Add `cuda-10.0/lib` to `$DYLD_LIBRARY_PATH` by doing
    
        $ export DYLD_LIBRARY_PATH=/usr/local/cuda-10.0/lib
        
    (**Windows**) TODO
    
        $ add to windows path
    
    Your machine should now be set up to work with CUDA. Troubleshooting associated with these steps can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions).
6. Create a conda environment from the `environment.yml` file by doing:
  
        $ conda env create -f environment.yml
          
    This will create an environment called `bionic` with all the required dependencies except for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).
7. Next, to install the required PyTorch Geometric dependencies on **Linux** run:

        $ . install_PyG.sh
        
    For **macOS** and **Windows**, do the following:
    1. TODO
    2. TODO
    3. TODO
    
8. Ensure the `bionic` environment is active and test BIONIC is properly installed with

        $ python main.py -c gav_krog.json
--->
## Usage

## Datasets
