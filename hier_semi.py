import numpy as np
import networkx as nx
import argparse
import math, random

import copy
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import DBSCAN

from Functions import image_array_list, generation_client_dictionary, init_list_of_objects
from Functions import stat_clients, distribution_generation, clustering_by_KMeans
from Functions import label_distance_matrix_generation_Hellinger, choose_repre_and_distance_matrix


from client_dsgd import ClientDSGD_Big

# We have 3 models. Each is used for one dataset.
class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class Net_FashionMNIST(nn.Module):
    def __init__(self):
        super(Net_FashionMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

## We use VGG11 for Cifar10 defautly.
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
## Set up the arguments

parser = argparse.ArgumentParser()
parser.add_argument("--Dataset",type=str,required=True)
parser.add_argument("--Partition",type=str,required=True)
parser.add_argument("--Epochs",type=int,default=201)
parser.add_argument("--Num_Client",type=int,default=50)
parser.add_argument("--Num_Cluster",type=int,default=10)
parser.add_argument("--Num_Neighbor",type=int,default=2)
parser.add_argument("--Intra_Interval",type=int,default=2)
parser.add_argument("--Inter_Interval",type=int,default=2)
parser.add_argument("--Test_Interval",type=int,default=1)
parser.add_argument("--Topology",type=str,required=True)
parser.add_argument("--Framework",type=str,required=True)
parser.add_argument("--Local_Epochs",type=int,default=50)
args = parser.parse_args()

dataset = args.Dataset
partition_method = args.Partition
cluster_number = args.Num_Cluster
client_number = args.Num_Client
learning_rate = 0.01
momentum = 0.5
test_interval = args.Test_Interval
n_epochs = args.Epochs
num_neighbors = args.Num_Neighbor
intra_interval = args.Intra_Interval
inter_interval = args.Inter_Interval
local_epoch = args.Local_Epochs

if dataset == "MNIST":
    
    ## For train data
    train_data = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                  transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))
                                  ]))
    
    train_data_image_array_list, train_data_label_array_list = image_array_list(train_data)
    
    
    ## For test data    
    test_data = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                  transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                      (0.1307,), (0.3081,))
                                  ]))
    
    test_data_image_array_list, test_data_label_array_list = image_array_list(test_data)

elif dataset == "FashionMNIST":
    
    ## For train data     
    train_data = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                    torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    train_data_image_array_list, train_data_label_array_list = image_array_list(train_data)
    
    ## For test data   
    test_data = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                                   torchvision.transforms.Compose([torchvision.transforms.ToTensor()])) 

    test_data_image_array_list, test_data_label_array_list = image_array_list(test_data)

elif dataset == "Cifar10":
    
    ## For train 
    train_data = torchvision.datasets.CIFAR10(root='./data/', train=True,download=True, 
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToTensor(), 
                                                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
    train_data_image_array_list, train_data_label_array_list = image_array_list(train_data)
    
    ## For test data    
    test_data = torchvision.datasets.CIFAR10(root='./data/', train=False,download=True, 
                                            transform=torchvision.transforms.Compose(
                                                [torchvision.transforms.ToTensor(), 
                                                 torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ]))
    test_data_image_array_list, test_data_label_array_list = image_array_list(test_data)
    
else:
    print("Unlucky your expectation is beyond the abilitiy of our code. Please use one of the three datasets or add the code here.")

## Generate a empty data dictionary

client_id_list, TrainDataDict = generation_client_dictionary(client_number)

## For each partition method:
    
if partition_method == "k-Means":
    ## Divide the data into clusters according to KMeans

    TrainDataDict, cluster_idx_for_all = clustering_by_KMeans(TrainDataDict, client_number, train_data_image_array_list,
                                              train_data_label_array_list, 0)

    client_list_for_dataloader = []
    for i in range(client_number):
        
        client_list_for_dataloader.append(torch.utils.data.Subset(train_data, cluster_idx_for_all[i]))
        
    ## Get the statistics, distribution, distance matrix of clients

    stat, client_sample_num = stat_clients(TrainDataDict, client_number)
    label_distribution, conditional_distribution = distribution_generation(client_number, stat,
                                                                            client_sample_num,TrainDataDict)
    label_distance_matrix = label_distance_matrix_generation_Hellinger(label_distribution)

    ## cluster the clients into 10 clusters
    eps = 0.26
    db = DBSCAN(eps=eps, min_samples=3, metric='precomputed').fit(label_distance_matrix)
    while db.labels_.min()<0:
        db.labels_ += 1

    new_dis_mat_idx, client_repre, new_client_with_order, new_cluster_idx, repre_topo_mat, new_client_list_for_dataloader = choose_repre_and_distance_matrix(client_list_for_dataloader, TrainDataDict, client_number, 
                                         client_id_list, db.labels_)

elif partition_method == "2-Label":
    index_list = list()
    Client_DataDict = {}
    label_id_list = [i for i in range(10)]
    client_data_list = [i for i in range(client_number)]
    for label_id in label_id_list:
        index_list.append(list())

    ## Save the idx of each label of train data  
    for idx in range(len(train_data_label_array_list)):

        sample = {}
        sample['y'] = train_data_label_array_list[idx]
        sample['x'] = train_data_image_array_list[idx]
        index_list[train_data_label_array_list[idx]].append(idx)

    ## Divide the train data into 100 parts, each label 10 parts

    data_division = []
    for label_id in range(10):
        for partition_id in range(10):
            partition_size = math.floor(len(index_list[label_id])/10)
            data_division.append(index_list[label_id][partition_id*partition_size:(partition_id+1)*partition_size])

    ## Merge data_partition into 50 clients

    index_2Labels = init_list_of_objects(50)
    for label_id in range(10):
        for client_id in range(5):
            index_2Labels[5*label_id+client_id]=copy.deepcopy(data_division[10*label_id+client_id])

    b = list(range(10))
    for i,j in enumerate(b):
        for client_id in range(5):
            index_2Labels[5*i+client_id]+=data_division[10*j+5+client_id]
            
    client_list_for_dataloader = []        

    for client_idx in range(client_number):
        client_list_for_dataloader.append(torch.utils.data.Subset(train_data, index_2Labels[client_idx]))
    ## Generate topology and representative matrix and client list according to this
    new_idx_for_clients = []
    new_client_list_for_dataloader = []
    for i in range(10):
        new_idx_for_clients.append(5*i)
    for i in range(10):
        for j in range(4):
            new_idx_for_clients.append(5*i+j+1)

    for i in range(50):
        
        new_client_list_for_dataloader.append(client_list_for_dataloader[new_idx_for_clients[i]])
        
    repre_topo_mat = np.zeros((10, client_number))
    np.fill_diagonal(repre_topo_mat,1)
    for i in range(10):
        repre_topo_mat[i,(10+4*i):(14+4*i)] = 1
     
## Dataloader
train_loader_all_clients = [torch.utils.data.DataLoader(x, batch_size=30, shuffle=True) for x in new_client_list_for_dataloader]
test_loader_all_clients = torch.utils.data.DataLoader(test_data, batch_size=30,shuffle=True)

## Backbone topology

if args.Topology == "Wheel":
    
    # Wheel(Based on topology)
    topology_generated = np.zeros((cluster_number,cluster_number))
    
    np.fill_diagonal(topology_generated,1)
    
    for i in range(0,cluster_number//2):
        topology_generated[i,i+cluster_number//2] = 1
        topology_generated[i+cluster_number//2,i] = 1
        
    
    for i in range(cluster_number):
        if i < cluster_number-1:
            topology_generated[i,i+1] = 1
        else:
            topology_generated[i,0] = 1

elif args.Topology == "Ring":
    # 2Ring
    topology_generated = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(cluster_number, 2, 0)), dtype=np.float32)
    np.fill_diagonal(topology_generated, 1)

elif args.Topology == "FC":
    # Fully
    topology_generated = np.ones((cluster_number,cluster_number))

elif args.Topology =="NotConnected":
    topology_generated = np.eye(cluster_number)
    
## Generate clients and models
if dataset == "MNIST":
    initial_model = Net_MNIST()
    client_models = [Net_MNIST() for _ in range(client_number)] 
elif dataset == "FashionMNIST":
    initial_model = Net_FashionMNIST()
    client_models = [Net_FashionMNIST() for _ in range(client_number)] 
elif dataset == "Cifar10":
    initial_model = VGG("VGG11")
    client_models = [VGG("VGG11") for _ in range(client_number)] 
for model in client_models:
    model.load_state_dict(initial_model.state_dict())
client_list = []

if partition_method == "k-Means":
    
    for i in range(client_number):
        
        if i < len(client_repre):           
            client_list.append(ClientDSGD_Big(i, train_loader_all_clients[i], test_loader_all_clients, topology_generated, repre_topo_mat, client_models[i], i, local_epoch, learning_rate, momentum))           
        else:
            client_list.append(ClientDSGD_Big(i, train_loader_all_clients[i], test_loader_all_clients, topology_generated, repre_topo_mat, client_models[i], new_cluster_idx[i], local_epoch, learning_rate, momentum))

elif partition_method == "2-Label":
    
    for i in range(client_number):
        
        if i < cluster_number:          
            client_list.append(ClientDSGD_Big(i, train_loader_all_clients[i], test_loader_all_clients, topology_generated, repre_topo_mat, client_models[i], i, local_epoch, learning_rate, momentum))            
        else:
            client_list.append(ClientDSGD_Big(i, train_loader_all_clients[i], test_loader_all_clients, topology_generated, repre_topo_mat,  client_models[i], (i-10)//4, local_epoch, learning_rate, momentum))
        
## Conduct training

## Make the test accuracy and test loss matrix, then Plot
train_loss_mat = np.zeros((client_number, n_epochs))
test_loss_mat = np.zeros((client_number, n_epochs//test_interval +1))
test_accuracy_mat = np.zeros((client_number, n_epochs//test_interval +1))
test_accuracy_before_communi_mat = np.zeros((client_number, n_epochs))
for idx in range(0, n_epochs):

    num_communication = 0
    num_within_communication = 0
    num_between_communication = 0
    # test_corrections_within_epoch = 0
    
    ## Training for each client
    for i in range(client_number):
        
        client_list[i].train_my()
        
        ##  Fill in the train_loss
        
        train_loss_mat[i,idx] = client_list[i].train_loss_epochs[-1]


        
    ## Within cluster's communication
    
    if idx % intra_interval == 1 and args.Framework == "Hierarchical":
        ## Communication from clients to representative 
        for i in range(len(new_dis_mat_idx), client_number):  
            client_list[i].send_local_gradient_to_representative(client_list)
        
        ## representative's side: Local update and send parameters back
        for i in range(len(new_dis_mat_idx)):
            
            client_list[i].update_local_parameters_within_as_representative()
            client_list[i].send_local_gradient_to_clients(client_list)
        ## Clients' side: Local update
        for i in range(len(new_dis_mat_idx), client_number):  
            client_list[i].update_local_parameters_within_as_clients()
    
    ## Between clusters' communication
    if idx % inter_interval == 0 and args.Topology != "NotConnected":
        
        for _ in range(1):
            
            for i in range(len(new_dis_mat_idx)):
                client_list[i].send_local_gradient_to_neighbor(client_list)
            for i in range(len(new_dis_mat_idx)):
                client_list[i].update_local_parameters_between()
        for i in range(len(new_dis_mat_idx)):            
            client_list[i].send_local_gradient_to_clients(client_list)
        ## Clients' side: Local update
        for i in range(len(new_dis_mat_idx), client_number):  
            client_list[i].update_local_parameters_within_as_clients()
        
    ## Calculate the number of communications
    for i in range(client_number):
        
        num_within_communication += client_list[i].num_within_communication
        num_between_communication += client_list[i].num_between_communication
    num_communication = num_within_communication + num_between_communication
    
    ## Test
    if idx % test_interval == 0 :
        
        for i in range(client_number):
            
            client_list[i].test_my(idx+1)
            test_loss_mat[i,idx//test_interval] = client_list[i].test_loss_epochs[-1]
            test_accuracy_mat[i,idx//test_interval] = client_list[i].test_accuracy_epochs[-1]