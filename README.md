### Introduction
BIONIC is a deep-learning based biological network integration algorithm that uses graph convolutional networks (GCN) to
learn integrated features for genes or proteins across input networks. BIONIC produces high-quality node features and is
scalable both in number of networks and network size.

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

### Installation
BIONIC is implemented in Python 3.7 and uses PyTorch.



### Usage

### Datasets
