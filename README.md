## :boom: Introduction
BIONIC (**Bio**logical **N**etwork **I**ntegration using **C**onvolutions) is a deep-learning based biological network integration algorithm that extends the graph convolutional network (GCN) to learn integrated features for genes or proteins across input networks. BIONIC produces high-quality gene features and is scalable both in number of networks and network size.

An overview of BIONIC can be seen below.

<p align="center">
  <a href="https://ibb.co/gFJbcCg"><img src="https://i.ibb.co/x86KrQ5/fig1a-nolabel.png" alt="fig1a-nolabel" border="0"></a>
</p>

1. Multiple networks are input into BIONIC
2. Each network is passed through its own graph convolutional encoder where network-specific gene features are learned based the network topologies. These features are summed to produce integrated gene features which capture salient topological information across input networks. The integrated features can then be used for downstream tasks, such as gene-gene functional linkage prediction, module detection (clustering) and gene function prediction.
3. In order to train and optimize the integrated gene features, BIONIC first decodes the integrated features into a reconstruction of the input networks.
4. BIONIC then minimizes the difference between this reconstruction and the input networks (i.e. reconstruction error) by updating its weights to learn gene features that capture relevant topological information.

## :gear: Installation
- BIONIC is implemented in [Python 3.8](https://www.python.org/downloads/) and uses [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

- BIONIC can run on the CPU or GPU. The CPU distribution will get you up and running quickly, but the GPU distributions are significantly faster for large models.

- Currently, we provide wheels for CPU, CUDA 9.2, CUDA 10.1 and CUDA 10.2 on Linux, and CPU, CUDA 10.2 and CUDA 10.2 on Windows.

**NOTE:** If you run into any problems with installation, please don't hesitate to open an [issue](https://github.com/bowang-lab/BIONIC/issues).

### Preinstallation for CUDA capable BIONIC

If you are installing a CUDA capable BIONIC wheel (i.e. not CPU), first ensure you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. Then, if you don't have CUDA installed and configured on your system already, [download](https://developer.nvidia.com/cuda-toolkit), install and configure a BIONIC compatible CUDA version. Nvidia provides detailed instructions on how to do this for both [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). 

### Installing from wheel (recommended)

1. Before installing BIONIC, it is recommended you create a virutal Python **3.8** environment using tools like the built in `venv` command, or [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/).

2. Make sure your virtual environment is active, then install BIONIC by running

       $ pip install bionic_model==${VERSION}+${CUDA} -f https://bowang-lab.github.io/BIONIC/wheels.html

where `${VERSION}` is the version of BIONIC you want to install (currently `0.1.0`) and `${CUDA}` is one of `cpu`, `cu92`, `cu101`, `cu102`, corresponding to the CPU, CUDA 9.2, CUDA 10.1 and CUDA 10.2 versions, respectively. Note, as above, that `cu92` is **not** available on Windows.

3. Test BIONIC is installed properly by running

       $ bionic --help
       
    You should see a help message. 

### Installing using Poetry

1. If you don't already have it, [install Poetry](https://python-poetry.org/docs/#installation).

2. Create a virtual Python **3.8** environment using tools like the built in `venv` command, or [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/). Make sure your virutal environment is active for the following steps.

3. Install PyTorch **1.6.0** for your desired CUDA version [here](https://pytorch.org/get-started/locally/).

4. Install PyTorch 1.6.0 compatible PyTorch Geometric dependencies for your desired CUDA version [here](https://github.com/rusty1s/pytorch_geometric#pytorch-160).

5. Make sure you are in the BIONIC root directory and run

       $ poetry install
       
6. Test BIONIC is installed properly by running

       $ bionic --help
       
    You should see a help message.

## :zap: Usage

### Configuration File
BIONIC runs by passing in a configuration file - a [JSON](https://www.w3schools.com/whatis/whatis_json.asp) file containing all the relevant model file paths and hyperparameters. You can have a uniquely named config file for each integration experiment you want to run. An example config file can be found [here](https://github.com/bowang-lab/BIONIC/blob/master/bionic/config/costanzo_hu_krogan.json).

The configuration keys are as follows:

Argument | Default | Description
--- | :---: | ---
`names` | N/A | Filenames of input networks. These files should be stored in `src/inputs`. By specifying `"*"` BIONIC will integrate all networks in `src/inputs`.
`out_name` | config file path | Path to prepend to all output files. If not specified it will be the path of the config file. `out_name` takes the format `path/to/output` where `output` is an extensionless output file name.
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
- The reconstruction loss may look like it's bottoming out early on but the model will continue improving feature quality for an unintuitively long time afterward.
- `embedding_size` directly affects the quality of learned features. We found the default `512` works for most networks, though it's worth experimenting with different sizes for your application. In general, higher `embedding_size` will encode more information present in the input networks but at the risk of also encoding noise.
- `gat_shapes.dimension` should be increased for networks with many nodes. We found `128` - `256` is a good size for human networks, for example.

#### Input Networks
- BIONIC runs faster and performs better with sparser networks - as a general rule, try to keep the average node degree below 50 for each network.

## :file_folder: Datasets
Supplementary files can be found [here](https://data.wanglab.ml/BIONIC/).
