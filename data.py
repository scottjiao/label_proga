import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from    scipy.sparse.linalg.eigen.arpack import eigsh
import  sys
import time
import torch
import os
import json

def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



class json_data_io():

    def __init__(self,file_name='temp'):
        self.file_name='{}.json'.format(file_name)

    def save(self,data_dict):
        with open(self.file_name,'w') as f:
            json.dump(data_dict,f)
    
    def load(self):
        with open(self.file_name,'r') as f:
            data_dict=json.load(f)
        return data_dict


def few_labels(adj, features, labels ,args):
    """
    convert the dataset into few-label data
    """

    ''' totalize the dataset and set the random seed for choices of labels '''
    # incoporate the test and valid set into training set
    y_total=labels
    #print(np.sum(np.sum(y_train)))
    #print(np.sum(np.sum(y_val)))
    #print(np.sum(np.sum(y_test)))

    # get the number of classes and samples
    number_samples,number_class=y_total.shape
    # set the random seed of choices for few label labels
    if args.few_label_seed:     # 
        np.random.seed(args.few_label_seed)
    else:
        np.random.seed(int(time.time()))  # if args.few_label_seed is equal to 0, set it as random seed, by the time function. 

    
    '''collect the training samples belonging to this class'''
    # initialize the collection set, each pot corresponds to a class
    collection=[]  
    for j in range(number_class): #
        collection.append([]) 
    # Check each node and put them into the corresponding pot of collection set
    for i in range(number_samples): 
        # get the class of this node
        temp_list=[int(i) for i in list(y_total[i])]
        if 1 in temp_list:
            temp_class=[int(i) for i in list(y_total[i])].index(1)
        else:
            temp_class=0  # give a non-sense number
        # put the id of this node into the corresponding pot 
        collection[temp_class].append(i)
    
    '''execute the random choice of few labels'''
    choosed_list=[]
    for j in range(number_class):
        # choose few labels from this set, by id.
        temp_list=collection[j]   
        if args.few_label_number>=len(temp_list):
            chosen_size=len(temp_list)
        else:
            chosen_size=args.few_label_number
        choosed_list.extend(np.random.choice(temp_list,size=chosen_size,replace=False))
    
    '''ids to matrix'''
    idx_test = sorted([i for i in list(range(number_samples)) if i not in choosed_list])
    idx_train = sorted(choosed_list)
    idx_val = []

    train_mask = sample_mask(idx_train, number_samples)
    val_mask = sample_mask(idx_val, number_samples)
    test_mask = sample_mask(idx_test, number_samples)

    y_train = np.zeros(y_total.shape)
    y_val = np.zeros(y_total.shape)
    y_test = np.zeros(y_total.shape)
    y_train[train_mask, :] = y_total[train_mask, :]
    y_val[val_mask, :] = y_total[val_mask, :]
    y_test[test_mask, :] = y_total[test_mask, :]
    
    



    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



def load_ogb_data(dataset_str):

    import ogb
    from ogb.nodeproppred.dataset import NodePropPredDataset

    dataset=NodePropPredDataset(name =dataset_str)

    graph, raw_labels=dataset[0]


    labels=torch.zeros((len(raw_labels),int(max(raw_labels))+1)).scatter_(1,torch.from_numpy( raw_labels),1)
    labels=labels.numpy()

    features=graph['node_feat']
    edge_lists=graph['edge_index']
    edge_lists=[ (edge_lists[0][i],edge_lists[1][i])   for i in range(len(edge_lists[0]))]
    adj=nx.adjacency_matrix(nx.from_edgelist(edge_lists))


    print(labels.shape)
    print(adj.shape)
    print(features.shape)
    
    idx_split=dataset.get_idx_split()

    #raise Exception

    idx_test = idx_split['test']
    idx_train = idx_split['train']
    idx_val = idx_split['valid']

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    #raise Exception


    return adj, features,labels, y_train, y_val, y_test, train_mask, val_mask, test_mask






def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #raise Exception
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    #raise Exception()
    return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features,sparsity=True):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    if sparsity:

        return sparse_to_tuple(features) # [coordinates, data, shape], []
    else:
        return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)





def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
