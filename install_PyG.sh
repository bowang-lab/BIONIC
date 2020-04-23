#!/bin/bash

conda activate bionic
pip install torch_cluster==1.4.2 --no-cache-dir
pip install torch_scatter==1.2.0 --no-cache-dir
pip install torch-sparse==0.4.0 --no-cache-dir
pip install git+https://github.com/duncster94/pytorch_geometric.git@master#egg=torch-geometric
