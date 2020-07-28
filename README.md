## Introduction
BIONIC (**Bio**logical **N**etwork **I**ntegration using **C**onvolutions) is a deep-learning based biological network integration algorithm that extends the graph convolutional network (GCN) to learn integrated features for genes or proteins across input networks. BIONIC produces high-quality gene features and is scalable both in number of networks and network size.

An overview of BIONIC can be seen below.

<p align="center">
  <a href="https://ibb.co/gFJbcCg"><img src="https://i.ibb.co/x86KrQ5/fig1a-nolabel.png" alt="fig1a-nolabel" border="0"></a>
</p>

1. Multiple networks are input into BIONIC
2. Each network is passed through its own graph convolutional encoder where network-specific gene features are learned based the network topologies. These features are summed to produce integrated gene features which capture salient topological information across input networks. The integrated features can then be used for downstream tasks, such as gene-gene functional linkage prediction, module detection (clustering) and gene function prediction.
3. In order to train and optimize the integrated gene features, BIONIC first decodes the integrated features into a reconstruction of the input networks.
4. BIONIC then minimizes the difference between this reconstruction and the input networks (i.e. reconstruction error) by updating its weights to learn gene features that capture relevant topological information.

BIONIC is implemented in [Python 3](https://www.python.org/downloads/) and uses [PyTorch](https://pytorch.org/).

## Installation
**NOTE: Currently BIONIC requires an NVIDIA GPU capable of supporting CUDA 10.0 to run.**

**Preinstallation:** First, ensure the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date.

### [Docker](https://www.docker.com/) (Recommended, Linux only)

If you are on a Linux machine it's recommended to run BIONIC in a Docker container.

1. Ensure your Docker version is at least 19.03.

2. Copy or download the Dockerfile from [here](https://raw.githubusercontent.com/bowang-lab/BIONIC/master/Dockerfile) by running

        $ wget https://raw.githubusercontent.com/bowang-lab/BIONIC/master/Dockerfile

3. Build the BIONIC Docker image by running

        $ docker build -t "bionic" /path/to/Dockerfile
   
   NOTE: This may take some time.
4. Install `nvidia-container-toolkit` by running

        $ apt-get install -y nvidia-container-toolkit
        
5. Create a BIONIC instance by running

        $ docker run -it --gpus all --ipc=host --network=host --volume=$PWD:/app -e NVIDIA_VISIBLE_DEVICES=0 "bionic" /bin/bash

6. (optional) Test BIONIC works by running the following inside the Docker container

        $ python main.py -c gav_krog.json

### Direct Installation (Linux and macOS)

1. Download and install [Anaconda](https://www.anaconda.com/products/individual) for Python 3.x if you have not done so already.

2. Install CUDA Toolkit 10.0 from [here](https://developer.nvidia.com/cuda-10.0-download-archive). NOTE: The CUDA Toolkit version **must be 10.0**, other versions (such as 10.1+) will not work.

3. Locate the CUDA Toolkit installation directory (it should be something similar to `/usr/local/cuda-10.0/bin`). Add this path to the `$PATH` variable by doing
        
        $ export PATH=/usr/local/cuda-10.0/bin:$PATH

4. Add `cuda-10.0/include` to `$CPATH` by running

        $ export CPATH=/usr/local/cuda-10.0/include:$CPATH
        
5. (**Linux**) Add `cuda-10.0/lib64` to `$LD_LIBRARY_PATH` by running

        $ export LD_LIBRARY_PATH=/usr/local/cuda-10.0/bin:$LD_LIBRARY_PATH

    (**macOS**) Add `cuda-10.0/lib` to `$DYLD_LIBRARY_PATH` by doing
    
        $ export DYLD_LIBRARY_PATH=/usr/local/cuda-10.0/lib
    
    Your machine should now be set up to work with CUDA. Troubleshooting associated with these steps can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions).
6. Clone this repository and navigate into the root directory

        $ git clone https://github.com/bowang-lab/BIONIC.git
        $ cd BIONIC

7. Create a conda environment from the `environment.yml` file by doing
  
        $ conda env create -f environment.yml
          
    This will create an environment called `bionic` with all the required dependencies except for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).
    **Ensure the** `bionic` **environment is active for the next step.**

8. Install the required PyTorch Geometric dependencies

        $ pip install torch-cluster==1.4.2 --no-cache-dir
        $ pip install torch-sparse==0.4.0 --no-cache-dir
        $ pip install torch-scatter==1.2.0 --no-cache-dir
        $ pip install git+https://github.com/duncster94/pytorch_geometric.git@master#egg=torch-geometric
    
9. (optional) Test BIONIC is properly installed with

        $ python main.py -c gav_krog.json

## Usage

### Configuration File
BIONIC runs by passing in a configuration file - a [JSON](https://www.w3schools.com/whatis/whatis_json.asp) file containing all the relevant model file paths and hyperparameters. You can have a uniquely named config file for each integration experiment you want to run. These config files are stored in the `src/config` directory where an example config file `gav_krog.json` is already present. `gav_krog.json` specifies the relevant parameters to integrate two, large-scale yeast protein-protein interaction networks - namely [Gavin et al. 2006](https://pubmed.ncbi.nlm.nih.gov/16429126/) and [Krogan et al. 2006](https://pubmed.ncbi.nlm.nih.gov/16554755/).

The configuration keys are as follows:

Argument | Default | Description
--- | :---: | ---
`names` | N/A | Filenames of input networks. These files should be stored in `src/inputs`. By specifying `"*"` BIONIC will integrate all networks in `src/inputs`.
`out_name` | config file name | Name to prepend to all output files. If not specified it will be the name of the config file.
`delimiter` | `" "` | Delimiter for input network files.
`epochs` | `3000` | Number of training steps to run BIONIC for (see [**usage tips**](#usage-tips)).
`batch_size` | `2048` | Number of genes in each mini-batch. Higher numbers result in faster training but also higher memory usage.
`sample_size` | `0` | Number of networks to batch over (`0` indicates **all** networks will be in each mini-batch). Higher numbers (or `0`) result in faster training but higher memory usage.
`learning_rate` | `0.0005` | Learning rate of BIONIC. Higher learning rates result in faster convergence but run the risk of unstable training (see [**usage tips**](#usage-tips)).
`embedding_size` | `512` | Dimensionality of the learned integrated gene features (see [**usage tips**](#usage-tips)).
`svd_dim` | `0` | Dimensionality of initial network features singular value decomposition (SVD) approximation. `0` indicates SVD is not applied. Setting this to `1024` or `2048` can be a useful way to speed up training and reduce memory consumption (especially for integrations with many genes) while incurring a small reduction in feature quality.
`initialization` | `"xavier"` | Weight initialization scheme. Valid options are `"xavier"` or `"kaiming"`.
`gat_shapes.dimension` | `64` | Dimensionality of each individual graph attention layer (GAT) head (see [**usage tips**](#usage-tips)).
`gat_shapes.n_heads` | `10` | Number of attention heads for each network-specific GAT.
`gat_shapes.n_layers` | `2` | Number of times each network is passed through its corresponding GAT. This number corresponds to the effective neighbourhood size of the convolution.
`save_network_scales` | `false` | Whether to save the internal learned network features scaling coefficients.
`save_model` | `true` | Whether to save the trained model parameters and state.
`use_tensorboard` | `false` | Whether to output training data and feature embeddings to Tensorboard. NOTE: Tensorboard is not included in the default installation and must be installed seperately.
`plot_loss` | `true` | Whether to plot the model loss curves after training.

By default, only the `names` key is required, though it is recommended you experiment with different hyperparameters so BIONIC suits your needs.

### Network Files

Input networks are text files in **edgelist** format, where each line consists of two gene identifiers and (optionally) the weight of the edge between them, for example:

```
geneA geneB 0.8
geneA geneC 0.75
geneB geneD 1.0
```

If the edge weight column is omitted, the network is considered binary (i.e. all edges will be given a weight of 1). These network files are stored in the `src/inputs` directory. The gene indentifiers and edge weights are delimited with spaces by default. If you have network files that use different delimiters, this can be specified in the config file by setting the `delimiter` key.
BIONIC assumes all networks are undirected and enforces this in its preprocessing step.

### Running BIONIC

To run BIONIC, do

    $ python main.py -c your_config_file.json

Results will be saved in the `src/outputs` directory.

### Usage Tips

The [configuration parameters table](#configuration-file) provides usage tips for most parameters. Additional suggestions are listed below. If you have any questions at all, please open an issue.

#### Hyperparameter Choice
- `learning_rate` and `epochs` have the largest effect on training time and performance. 
- `learning_rate` should generally be reduced as you integrate more networks. If the model loss suddenly increases by an order of magnitude or more at any point during training, this is a sign `learning_rate` needs to be lowered.
- `epochs` should be increased as you integrate more networks. 10000-15000 epochs is not unreasonable for 50+ networks.
- The reconstruction loss may look like it's bottoming out early on but the model will continue improving feature quality for an unintuitively long time afterward.
- `embedding_size` directly affects the quality of learned features. We found the default `512` works for most networks, though it's worth experimenting with different sizes for your application. In general, higher `embedding_size` will encode more information present in the input networks but at the risk of also encoding noise.
- `gat_shapes.dimension` should be increased for networks with many nodes. We found `128` - `256` is a good size for human networks, for example.

#### Input Networks
- BIONIC runs faster and performs better with sparser networks - as a general rule, try to keep the average node degree below 50 for each network.

## Datasets
TODO
