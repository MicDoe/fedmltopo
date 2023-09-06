import numpy as np
from sklearn.cluster import KMeans
import cv2
        
# Transfer dataset object to numpy array

def image_array_list(data):
    
    data_image_list = []
    data_label_list = []
    
    # Transfer the data to numpy list
    for i in range(data.targets.size(0)):
        data_image_list.append(data[i][0][0].reshape(784,).numpy())
        data_label_list.append(int(data.targets[i].numpy()))
    
    # We need numpy array
    data_label_array_list = np.array(data_label_list)
    data_image_array_list = np.array(data_image_list)
        
    return  data_image_array_list, data_label_array_list

# Generate clients dictionary

def generation_client_dictionary(client_number):
    DataDict = {}
    client_id_list = [i for i in range(client_number)]
    for client_id in client_id_list:
        DataDict[client_id] = []
    return client_id_list, DataDict

# Generate a number of empty lists
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

# Splitting/clustering the data into #client_number sets

def clustering_by_KMeans(DataDict, client_number, data_image_array_list, 
                         data_label_array_list, random_state_number):
    
    kmeans = KMeans(n_clusters=client_number, random_state=random_state_number)
    kmeans.fit(data_image_array_list)
    clusters = kmeans.labels_
    cluster_idx_for_all_samples = init_list_of_objects(client_number)
    for i,cluster in enumerate(clusters):
        sample = {}
        sample['y'] = data_label_array_list[i]
        sample['x'] = data_image_array_list[i]
        DataDict[cluster].append(sample)
        cluster_idx_for_all_samples[cluster].append(i)
    return DataDict, cluster_idx_for_all_samples




# Return the statistics of the clients
def stat_clients(DataDict, client_number):    
    stat = []
    client_sample_num = []
    
    for i in range(client_number):
        
        num = np.zeros(10)
        
        for j in range(len(DataDict[i])):
            num[DataDict[i][j]['y']] += 1
            
        stat.append(num)
        client_sample_num.append(len(DataDict[i]))
    return stat, client_sample_num

# Generate the label and conditional distribution

def distribution_generation(client_number, stat, client_sample_num, DataDict):
    

    label_distribution = []
    conditional_distribution = []
    for i in range(client_number):
        label_distribution.append(stat[i]/client_sample_num[i])
    ## conditional distribution
    all_ones = np.ones([784,], dtype="float32")
    for i in range(client_number):
        within_client_stack = []
        counter_list = []
        idx = 0
        within_label_counter = 0
        within_label_accumulation = np.zeros((784, ), dtype="float32")
        for _, row in enumerate(stat[i]):
            within_label_accumulation = np.zeros((784, ), dtype="float32")
            
            within_label_counter = 0
            if int(row) == 0:
                within_client_stack.append(all_ones/len(all_ones))
                counter_list.append(0)
    ## iterate within the same label
            else:
                for j in range(int(row)):
                    within_label_counter += 1
                    within_label_accumulation += DataDict[i][idx]['x']
                    idx += 1
                feature_mean_before_distribution = within_label_accumulation/row
                ## ger rid of negative values
                feature_mean_before_distribution += np.abs(feature_mean_before_distribution.min())
                feature_mean = feature_mean_before_distribution / sum(feature_mean_before_distribution)
                within_client_stack.append(feature_mean)
                counter_list.append(within_label_counter)
        conditional_distribution.append(within_client_stack)
    return label_distribution, conditional_distribution

# Generte the label distance matrix
def label_distance_matrix_generation_Hellinger(distribution):
    
    distance_matrix = np.zeros([len(distribution),len(distribution)])
    for i in range(len(distribution)):
        for j in range(i, len(distribution)):
            distance_matrix[i, j] = H(distribution[i], distribution[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

# Hellinger Distance
def H(p, q):
  # distance between p an d
  # p and q are np array probability distributions
  n = len(p)
  sum = 0.0
  for i in range(n):
    sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
  result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
  return result



# Choose the representive of each cluster and calculate the distance matrix
     
def choose_repre_and_distance_matrix(Client_list_for_DataLoader, StreamingDataDict, client_number, 
                                     client_id_list, dbscan_labels):
    # See which cluser has which clients
    client_cluster_idx = []
    for i in range(dbscan_labels.max()+1):
        b = []
        for j in range(client_number):
            if dbscan_labels[j] == i:
                b.append(j)
        client_cluster_idx.append(b)
    
    client_family = {}
    for client_id in client_id_list:
        client_family[client_id] = []
        
    for i in range(client_number):
        temp_dict = {}
        temp_dict['size'] = len(StreamingDataDict[i])
        temp_dict['cluster_idx'] = dbscan_labels[i]
        client_family[i].append(temp_dict)
    
    client_repre = []
    new_dis_mat_idx = []
    for i in range(len(np.bincount(dbscan_labels))):
        if np.bincount(dbscan_labels)[i] == 1:
            for j in range(client_number):
                if client_family[j][0]['cluster_idx'] == i:
                    client_repre.append(StreamingDataDict[j])
                    new_dis_mat_idx.append(j)
        else:
            biggest_size = 0
            for j in range(client_number):
                if client_family[j][0]['cluster_idx'] == i:
                    if client_family[j][0]['size'] > biggest_size:
                        idx = j
                        biggest_size = client_family[j][0]['size']
            client_repre.append(StreamingDataDict[idx])
            new_dis_mat_idx.append(idx)
    
    # Clients according to new order and the new_cluster_idx indicates which cluster this belongs 
    # dataset_obj_list_with_new_order stands for the dataset obj list which is used for pytorch dataloader
    dataset_obj_list_with_new_order = []
    for i, idx in enumerate(new_dis_mat_idx):
        dataset_obj_list_with_new_order.append(Client_list_for_DataLoader[idx])
        
    new_client_with_order = client_repre.copy()
    new_client_with_order_idx = new_dis_mat_idx.copy()
    new_cluster_idx = []
    leftover_indicator = np.bincount(dbscan_labels) - 1
    for i in range(client_number):
        if i < len(new_dis_mat_idx):
            new_cluster_idx.append(i)
    
    for i in range(len(leftover_indicator)):
        if leftover_indicator[i] > 0:
            for j in range(leftover_indicator[i]):
                new_cluster_idx.append(i)
            for m in client_cluster_idx[i]:
                if (m not in new_dis_mat_idx):
                    new_client_with_order.append(StreamingDataDict[m])
                    dataset_obj_list_with_new_order.append(Client_list_for_DataLoader[m])
    
    # repre_topo_mat indicates which one is the representive in the cluster(2) and its buddies(1)
    repre_topo_mat = np.zeros((len(client_repre), client_number))
    numpy_new_cluster_idx = np.array(new_cluster_idx)
    for i in range(len(client_repre)):
        repre_topo_mat[i,i] = 2
        if len(np.where(np.array(new_cluster_idx) == i)[0]) > 1:
            for j in np.where(np.array(new_cluster_idx) == i)[0]:
                if j >= len(new_dis_mat_idx):
                    repre_topo_mat[i,j] = 1
    
    return new_dis_mat_idx, client_repre, new_client_with_order, new_cluster_idx, repre_topo_mat, dataset_obj_list_with_new_order

