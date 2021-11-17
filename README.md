![Build status](https://img.shields.io/github/workflow/status/bowang-lab/BIONIC/Python%20package)
![Version](https://img.shields.io/github/v/release/bowang-lab/BIONIC)
![Top language](https://img.shields.io/github/languages/top/bowang-lab/BIONIC)
![License](https://img.shields.io/github/license/bowang-lab/BIONIC)

**Check out the [preprint](https://www.biorxiv.org/content/10.1101/2021.03.15.435515v1)!**

## :boom: Introduction
BIONIC (**Bio**logical **N**etwork **I**ntegration using **C**onvolutions) is a deep-learning based biological network integration algorithm that incorporates graph convolutional networks (GCNs) to learn integrated features for genes or proteins across input networks. BIONIC produces high-quality gene features and is scalable both in the number of networks and network size.

An overview of BIONIC can be seen below.

<p align="center">
  <img src="https://raw.githubusercontent.com/bowang-lab/BIONIC/development/architecture_diagram.png" alt="BIONIC architecture diagram" border="0">
</p>

1. Multiple networks are input into BIONIC
2. Each network is passed through its own graph convolutional encoder where network-specific gene (node) features are learned based the network topologies. These features can be passed through the encoder multiple times to produce gene features which incorporate higher-order neighborhoods. These features are summed to produce integrated gene features which capture topological information across input networks. The integrated features can then be used for downstream tasks, such as gene co-annotation prediction, module detection (via clustering) and gene function prediction (via classification).
3. In order to train and optimize the integrated gene features, BIONIC first decodes the integrated features into a reconstruction of the input networks (**a**) and, if labelled data is available for some of the genes (such as protein complex membership, Gene Ontology annotations, etc.), BIONIC can also attempt to predict these functional labels (**b**). Note that any amount of labelled data can be used, from none (fully unsupervised), to labels for every gene, and everything in between.
4. BIONIC then minimizes the difference between the network reconstruction and the input networks (i.e. reconstruction error) by updating its weights to learn gene features that capture relevant topological information (**a**) and, if labelled data is provided, BIONIC updates its weights to minimizes the difference between the label predictions and true labels (**b**). 

## :gear: Installation
- BIONIC is implemented in [Python 3.8](https://www.python.org/downloads/) and uses [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

- BIONIC can run on the CPU or GPU. The CPU distribution will get you up and running quickly, but the GPU distributions are significantly faster for large models (when run on a GPU), and are recommended.

We provide wheels for the different versions of BIONIC, CUDA, and operating systems as follows:

**BIONIC 0.2.0 (Latest, Recommended)**

<i></i> | `cpu` | `cu92` | `cu101` | `cu102` | `cu111`
--- | --- | --- | --- | --- | ---
Linux | ✔️ |  |  | ✔️ | ✔️
Windows | ✔️ |  |  | ✔️ | ✔️

**BIONIC 0.1.0**

<i></i> | `cpu` | `cu92` | `cu101` | `cu102` | `cu111`
--- | --- | --- | --- | --- | ---
Linux | ✔️ | ✔️ | ✔️ | ✔️ | 
Windows | ✔️ |  | ✔️ | ✔️ | 


**NOTE:** If you run into any problems with installation, please don't hesitate to open an [issue](https://github.com/bowang-lab/BIONIC/issues).

### Preinstallation for CUDA capable BIONIC

If you are installing a CUDA capable BIONIC wheel (i.e. not CPU), first ensure you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. Then, if you don't have CUDA installed and configured on your system already, [download](https://developer.nvidia.com/cuda-toolkit), install and configure a BIONIC compatible CUDA version. Nvidia provides detailed instructions on how to do this for both [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). 

### Installing from wheel (recommended for general use)

1. Before installing BIONIC, it is recommended you create a virutal Python **3.8** environment using tools like the built in `venv` command, or [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/).

2. Make sure your virtual environment is active, then install BIONIC by running

       $ pip install https://github.com/bowang-lab/BIONIC/releases/download/v${VERSION}/bionic_model-${VERSION}+${CUDA}-cp38-cp38-${OS}.whl

    where `${VERSION}`, `${CUDA}` and `${OS}` correspond to the BIONIC version (latest is `0.2.0`), valid CUDA version (as specified above), and operating system, respectively. `${OS}` takes a value of `linux_x86_64` for Linux, and `win_amd64` for Windows. 
    
    For example, if we wanted to install the latest version of BIONIC to run on the CPU on a Linux system, we would run
    
       $ pip install https://github.com/bowang-lab/BIONIC/releases/download/v0.2.0/bionic_model-0.2.0+cpu-cp38-cp38-linux_x86_64.whl

    **NOTE:** There is a [known bug](https://github.com/pypa/pip/issues/7626) in certain versions of `pip` which may result in a `No matching distribution` error. If this occurs, install `pip==19.3.1` and try again.

3. Test BIONIC is installed properly by running

       $ bionic --help
       
    You should see a help message. 

### Installing using Poetry (recommended for development)

1. If you don't already have it, [install Poetry](https://python-poetry.org/docs/#installation).

2. Create a virtual Python **3.8** environment using tools like the built in `venv` command, or [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/). Make sure your virutal environment is active for the following steps.

3. Install PyTorch **1.9.0** for your desired CUDA version as follows:

       $ pip install torch==1.9.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
       
    where `${CUDA}` is the one of the options listed in the table above.

4. Install PyTorch 1.9.0 compatible [PyTorch Geometric dependencies](https://github.com/rusty1s/pytorch_geometric#pytorch-190) for your desired CUDA version as follows:

       $ pip install torch-scatter==2.0.8 torch-sparse==0.6.11 torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
       $ pip install torch-geometric==1.7.2

    where `${CUDA}` is the one of the options listed in the table above.

5. Clone this repository by running

       $ git clone https://github.com/bowang-lab/BIONIC.git

6. Make sure you are in the root directory (same as `pyproject.toml`) and run

       $ poetry install
       
7. Test BIONIC is installed properly by running

       $ bionic --help
       
    You should see a help message.

## :zap: Usage

### Configuration File
BIONIC runs by passing in a configuration file: a [JSON](https://www.w3schools.com/whatis/whatis_json.asp) file containing all the relevant model file paths and hyperparameters. You can have a uniquely named config file for each integration experiment you want to run. An example config file can be found [here](https://github.com/bowang-lab/BIONIC/blob/master/bionic/config/yeast_gi_coex_ppi.json).

The configuration keys are as follows:

Argument | Default | Description
--- | :---: | ---
`net_names` | N/A | Filepaths of input networks. By specifying `"*"` after the path, BIONIC will integrate all networks in the directory.
`label_names` | N/A | Filepaths of node label JSON files. An example node label file can be found [here](https://github.com/bowang-lab/BIONIC/blob/master/bionic/inputs/yeast_IntAct_complex_labels.json).
`out_name` | config file path | Path to prepend to all output files. If not specified it will be the path of the config file. `out_name` takes the format `path/to/output` where `output` is an extensionless output file name.
`delimiter` | `" "` | Delimiter for input network files.
`epochs` | `3000` | Number of training steps to run BIONIC for (see [**usage tips**](#usage-tips)).
`batch_size` | `2048` | Number of genes in each mini-batch. Higher numbers result in faster training but also higher memory usage.
`sample_size` | `0` | Number of networks to batch over (`0` indicates **all** networks will be in each mini-batch). Higher numbers (or `0`) result in faster training but higher memory usage.
`learning_rate` | `0.0005` | Learning rate of BIONIC. Higher learning rates result in faster convergence but run the risk of unstable training (see [**usage tips**](#usage-tips)).
`embedding_size` | `512` | Dimensionality of the learned integrated gene features (see [**usage tips**](#usage-tips)).
`shared_encoder` | `false` | Whether to use the same graph attention layer (GAT) encoder for all the input networks. This may lead to better performance in certain circumstances.
`svd_dim` | `0` | Dimensionality of initial network features singular value decomposition (SVD) approximation. `0` indicates SVD is not applied. Setting this to `1024` or `2048` can be a useful way to speed up training and reduce memory consumption (especially for integrations with many genes) while incurring a small reduction in feature quality.
`initialization` | `"kaiming"` | Weight initialization scheme. Valid options are `"xavier"` or `"kaiming"`.
`lambda` | N/A | Relative weighting between reconstruction and classification loss: `final_loss = lambda * rec_loss + (1 - lambda) * cls_loss`. Only relevant if `label_names` is specified. If `lambda` is not provided but `label_names` is, `lambda` will deafult to `0.95`.
`gat_shapes.dimension` | `64` | Dimensionality of each individual GAT head (see [**usage tips**](#usage-tips)).
`gat_shapes.n_heads` | `10` | Number of attention heads for each network-specific GAT.
`gat_shapes.n_layers` | `2` | Number of times each network is passed through its corresponding GAT. This number corresponds to the effective neighbourhood size of the convolution.
`save_network_scales` | `false` | Whether to save the internal learned network features scaling coefficients.
`save_label_predictions` | `false` | Whether to save the predicted node labels (if applicable).
`save_model` | `true` | Whether to save the trained model parameters and state.
`use_tensorboard` | `false` | Whether to output training data and feature embeddings to Tensorboard. NOTE: Tensorboard is not included in the default installation and must be installed seperately.
`plot_loss` | `true` | Whether to plot the model loss curves after training.

By default, only the `net_names` key is required, though it is recommended you experiment with different hyperparameters to suit your needs.

### Network Files

Input networks are text files in **edgelist** format, where each line consists of two gene identifiers and (optionally) the weight of the edge between them, for example:

```
geneA geneB 0.8
geneA geneC 0.75
geneB geneD 1.0
```

If the edge weight column is omitted, the network is considered binary (i.e. all edges will be given a weight of 1). The gene indentifiers and edge weights are delimited with spaces by default. If you have network files that use different delimiters, this can be specified in the config file by setting the `delimiter` key.
BIONIC assumes all networks are undirected and enforces this in its preprocessing step.

### Running BIONIC

To run BIONIC, do

    $ bionic path/to/your_config_file.json

Results will be saved in the `out_name` directory as specified in the config file.

### Usage Tips

The [configuration parameters table](#configuration-file) provides usage tips for many parameters. Additional suggestions are listed below. If you have any questions at all, please open an [issue](https://github.com/bowang-lab/BIONIC/issues).

#### Hyperparameter Choice
- `learning_rate` and `epochs` have the largest effect on training time and performance. 
- `learning_rate` should generally be reduced as you integrate more networks. If the model loss suddenly increases by an order of magnitude or more at any point during training, this is a sign `learning_rate` needs to be lowered.
- `epochs` should be increased as you integrate more networks. 10000-15000 epochs is not unreasonable for 50+ networks.
- `embedding_size` directly affects the quality of learned features. We found the default `512` works for most networks, though it's worth experimenting with different sizes for your application. In general, higher `embedding_size` will encode more information present in the input networks but at the risk of also encoding noise.
- `gat_shapes.dimension` should be increased for networks with many nodes. We found `128` - `256` is a good size for human networks, for example.

#### Input Networks
- BIONIC runs faster and performs better with sparser networks - as a general rule, try to keep the average node degree below 50 for each network.
