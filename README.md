## Name
Communication Topologies for Decentralized Federated Learning

## Description
The code here is for the paper "Communication Topologies for Decentralized Federated Learning".

## Installation
For the implementation, environment is built in the following way:  

conda create --name Whatever python=3.7
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
conda install -c anaconda scikit-learn pyflakes seaborn networkx
conda install -c conda-forge wandb matplotlib
conda install -c plotly plotly
python -m pip install opencv-python

## Usage
For example,
There are 3 options for datasets: "MNIST", "FashionMNIST" and "Cifar10".  

There are 2 options for partition: "k-Means" and "2-Label"

python3 main_DFL.py --Dataset="MNIST" --Partition="k-Means" --Epochs=201 --Num_Client=50 --Num_Neighbor=2 --Inter_Interval=1 --Test_Interval=1 --Topology='FC' --Framework='DFL' --Local_Epochs=10  

python3 main_Hier_Semi.py --Dataset="MNIST" --Partition="k-Means" --Epochs=201 --Num_Client=50 --Num_Cluster=10 --Num_Neighbor=2 --Intra_Interval=1 --Inter_Interval=1 --Test_Interval=1 