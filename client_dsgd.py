import random, copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class ClientDSGD_Big(object):
    def __init__(self, client_id, train_data_loader, test_data_loader, topology_manager, within_cluster_topo, model, representative_id, iteration_number,
                 learning_rate, momentum):

        # logging.info("streaming_data = %s" % streaming_data)

        # Since we use logistic regression, the model size is small.
        # Thus, independent model is created each client.
        self.model = model

        # self.topology_manager = topology_manager
        self.id = client_id  # integer
        self.train_data = train_data_loader
        self.test_data = test_data_loader
        
        self.representative_id = representative_id
        
        ## Identify whether the client itself is representative or not
        
        if self.representative_id == self.id:
            self.is_representative = True
        ## Clients don't need this topo
        if self.id < topology_manager.shape[0]:
            
            self.between_topo = topology_manager[client_id]
            self.between_topo = self.between_topo/np.count_nonzero(self.between_topo)
            self.communi_topo = topology_manager[:,client_id]
        self.within_topo = within_cluster_topo[self.representative_id]
        
        self.num_samples = len(self.train_data.dataset)
        self.num_within_communication = 0
        self.num_between_communication = 0
        # if self.b_symmetric:
        #     self.topology = topology_manager.get_symmetric_neighbor_list(client_id)
        # else:
        #     self.topology = topology_manager.get_asymmetric_neighbor_list(client_id)
        # print(self.topology)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        self.learning_rate = learning_rate
        self.iteration_number = iteration_number
        # TODO:
        # self.latency = random.uniform(0, latency)
        self.train_loss_epochs = []
        self.train_loss = []
        
        self.test_loss_epochs = []
        self.test_accuracy_epochs = []
        # the default weight of the model is z_t, while the x weight is another weight used as temporary value
        self.model_x = copy.deepcopy(self.model)

        # neighbors_weight_dict: between means that between clusters
        self.neighbors_between_paras_dict = dict()
        self.neighbors_between_topo_weight_dict = dict()
        # within means that within clusters
        self.neighbors_within_paras_dict = dict()
        self.neighbors_within_topo_weight_dict = dict()
    
    def train_my(self):
        self.model.train()
        self.train_loss = []
        for batch_idx, (data, target) in enumerate(self.train_data):
            if batch_idx < self.iteration_number:              
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()            
                self.optimizer.step()
                # if batch_idx % log_interval == 0:
                #   print('Train Epoch : {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                #                                           100. * batch_idx / len(train_loader), loss.item()))
                self.train_loss.append(loss.item())
                # train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        self.train_loss_epochs.append(sum(self.train_loss)/self.num_samples)
    
    
    def test_my(self, n_epoch):
        
        self.model.eval()
        test_loss = 0
        self.correct = 0
        with torch.no_grad():
            for data, target in self.test_data:
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                self.correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(self.test_data.dataset)
            self.test_loss_epochs.append(test_loss)
            # print('\nTest client {} in Epoch {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #                             self.id, n_epoch, test_loss, self.correct, len(self.test_data.dataset),
            #                             100. * self.correct / len(self.test_data.dataset)))
            self.test_accuracy_epochs.append((self.correct / len(self.test_data.dataset)).numpy())
            
    def get_regret(self):
        return self.loss_in_each_iteration

    ## simulation between clusters
    
    def send_local_gradient_to_neighbor(self, client_list):
        self.model_x = copy.deepcopy(self.model)
        for index in range(len(self.communi_topo)):
            if self.communi_topo[index] != 0 and index != self.id:
                client = client_list[index]
                wei = client.between_topo.max()
                client.receive_neighbor_gradients(self.id, self.model_x, wei)
                self.num_between_communication += 1
    def receive_neighbor_gradients(self, client_id, model_x, topo_weight):
        self.neighbors_between_paras_dict[client_id] = copy.deepcopy(model_x)
        self.neighbors_between_topo_weight_dict[client_id] = topo_weight

    def update_local_parameters_between(self):

        # update x_{t+1/2}
        for x_paras in self.model_x.parameters():
            x_paras.data.mul_(self.between_topo[self.id])

        for client_id in self.neighbors_between_paras_dict.keys():
            model_x = self.neighbors_between_paras_dict[client_id]
            topo_weight = self.neighbors_between_topo_weight_dict[client_id]
            for x_paras, x_neighbor in zip(list(self.model_x.parameters()), list(model_x.parameters())):
                temp = x_neighbor.data.mul(topo_weight)
                x_paras.data.add_(temp)

        # update parameter z (self.model)
        for x_params, z_params in zip(list(self.model_x.parameters()), list(self.model.parameters())):
            z_params.data.copy_(x_params)
    def update_local_parameters_between_show(self):

        # update x_{t+1/2}
        for x_paras in self.model_x.parameters():
            x_paras.data.mul_(self.between_topo[self.id])
        print()
        for client_id in self.neighbors_between_paras_dict.keys():
            model_x = self.neighbors_between_paras_dict[client_id]
            topo_weight = self.neighbors_between_topo_weight_dict[client_id]
            for x_paras, x_neighbor in zip(list(self.model_x.parameters()), list(model_x.parameters())):
                temp = x_neighbor.data.mul(topo_weight)
                x_paras.data.add_(temp)

        # update parameter z (self.model)
        for x_params, z_params in zip(list(self.model_x.parameters()), list(self.model.parameters())):
            z_params.data.copy_(x_params)         
    ## simulation within clusters
    
    ## Send model parameters to representative from clients
    def send_local_gradient_to_representative(self, client_list):
        self.model_x = copy.deepcopy(self.model)

        client = client_list[self.representative_id]
        client.receive_clients_gradients(self.id, self.model_x, self.num_samples)
        self.num_within_communication += 1
        
    def receive_clients_gradients(self, client_id, model_x, topo_weight):
        self.neighbors_within_paras_dict[client_id] = model_x
        self.neighbors_within_topo_weight_dict[client_id] = topo_weight
    
    def update_local_parameters_within_as_representative(self):
        
        
        # calculate the num_cluster
        self.num_cluster = 0
        self.num_cluster += self.num_samples
        for client_id in self.neighbors_within_paras_dict.keys():
            self.num_cluster += self.neighbors_within_topo_weight_dict[client_id]
        # update x_{t+1/2}
        for x_paras in self.model_x.parameters():
            x_paras.data.mul_( self.num_samples / self.num_cluster)
        

        for client_id in self.neighbors_within_paras_dict.keys():
            model_x = self.neighbors_within_paras_dict[client_id]
            topo_weight = self.neighbors_within_topo_weight_dict[client_id] / self.num_cluster

            for x_paras, x_neighbor in zip(list(self.model_x.parameters()), list(model_x.parameters())):
                temp = x_neighbor.data.mul(topo_weight)            
                x_paras.data.add_(temp)

        # update parameter z (self.model)
        for x_params, z_params in zip(list(self.model_x.parameters()), list(self.model.parameters())):
            z_params.data.copy_(x_params)
    ## Send model parameters to clients from representative
    def send_local_gradient_to_clients(self, client_list):
        self.model_x = copy.deepcopy(self.model)
        for index in range(len(self.within_topo)):
            if self.within_topo[index] != 0 and index != self.id:
                client = client_list[index]
                client.receive_representative_gradients(self.id, self.model_x, 1)
                self.num_within_communication += 1
    def receive_representative_gradients(self, client_id, model_x, topo_weight):
        self.neighbors_within_paras_dict[client_id] = model_x
        self.neighbors_within_topo_weight_dict[client_id] = topo_weight

    def update_local_parameters_within_as_clients(self):
        # update x_{t+1/2}
        for x_paras in self.model_x.parameters():
            x_paras.data.mul_(0)

        for client_id in self.neighbors_within_paras_dict.keys():
            model_x = self.neighbors_within_paras_dict[client_id]
            topo_weight = 1
            for x_paras, x_neighbor in zip(list(self.model_x.parameters()), list(model_x.parameters())):
                temp = x_neighbor.data.mul(topo_weight)
                x_paras.data.add_(temp)

        # update parameter z (self.model)
        for x_params, z_params in zip(list(self.model_x.parameters()), list(self.model.parameters())):
            z_params.data.copy_(x_params)