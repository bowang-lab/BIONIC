import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="BioNIC (Biological Network Integration using Convolutions)",
    version="0.1.0",
    author="Duncan Forster",
    author_email="duncan.forster@mail.utoronto.ca",
    license="MIT",
    description="Accurate, scalable biological network integration using graph convolutional networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords=[
        "Biological Network Integration",
        "Protein-protein Interaction",
        "Co-expression",
        "Genetic Interaction"
        "Omics",
        "Graph Convolutional Networks",
        "Graph Neural Networks",
        "Graph Attention Network"
    ],
    install_requires=[
        "torch-geometric @ https://github.com/duncster94/pytorch_geometric/master.zip#egg=torch-geometric-1.0"
    ],
    zip_safe=False
)