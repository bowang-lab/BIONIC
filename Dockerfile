FROM continuumio/miniconda3
# FROM gcc

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN echo "source activate bionic-env" > ~/.bashrc
ENV PATH /opt/conda/envs/bionic-env/bin:$PATH

# RUN python -c "import torch; print(torch.__version__)"
# RUN python -c "import torch; print(torch.cuda.is_available())"
# RUN conda install git pip
# RUN pip install --no-cache-dir torch-scatter
# RUN pip install --no-cache-dir torch-sparse
# RUN pip install --no-cache-dir torch-cluster
# RUN pip install --no-cache-dir git+https://github.com/duncster94/pytorch_geometric@master