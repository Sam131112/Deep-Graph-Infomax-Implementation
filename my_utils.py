import pickle as pk
import numpy as np
import networkx as nx
import scipy as sp
import torch
import infomap


#ind.cora.allx  ind.cora.ally  ind.cora.graph  ind.cora.test.index  ind.cora.tx  ind.cora.ty  ind.cora.x  ind.cora.y

def get_test_index(f):
    f1 = open(f,"r")
    ids = []
    for _f in f1:
        _f = _f.strip("\n\r")
        ids.append(int(_f))
    _t = np.array(ids)
    #_t = np.sort(_t)
    return _t


def convert_adj_to_sparse(adj):
    sparse_adj = sp.sparse.coo_matrix(adj,dtype=np.float32)
    indices = np.vstack((sparse_adj.row,sparse_adj.col))
    values = sparse_adj.data
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = sparse_adj.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(s))



def get_data(cuda="True"):
    end = [".allx",".ally",".tx",".ty",".x",".y",".graph",".test.index"]
    _allx,_ally,_te_x,_te_y,_tr_x,_tr_y,_net,_test = [u[0]+u[1] for u in zip(["ind.cora"]*8,end)]  # Change the network here 
    allx = pk.load(open(_allx,"rb"),encoding="latin1")
    ally = pk.load(open(_ally,"rb"),encoding="latin1")
    te_x = pk.load(open(_te_x,"rb"),encoding="latin1")
    te_y = pk.load(open(_te_y,"rb"),encoding="latin1")
    tr_x = pk.load(open(_tr_x,"rb"),encoding="latin1")
    tr_y = pk.load(open(_tr_y,"rb"),encoding="latin1")
    net = pk.load(open(_net,"rb"),encoding="latin1")
    test_idx = get_test_index(_test)
    test_idx_reorder = np.sort(test_idx)
    #print(tr_x.shape,te_x.shape,test_id.size,allx.shape)
    features = sp.sparse.vstack((allx,te_x)).tolil()  # because without lil format there is computational burden
    print(features.shape)
    features[test_idx,:] = features[test_idx_reorder,:]
    labels = np.vstack((ally,te_y))
    labels[test_idx,:] = labels[test_idx_reorder,:]
    idx_train = range(len(tr_y))
    idx_val = range(len(tr_y), len(tr_y)+500)
    g = nx.from_dict_of_lists(net)
    print("number of nodes,edges ",g.number_of_nodes(),g.number_of_edges())
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])   # A = A'
    #print(adj[5,2546],adj[2546,5])
    #adj = sp.sparse.coo_matrix(adj)
    adj = convert_adj_to_sparse(adj)
    adj = normalize_adj(adj)            # make this commented for GraphSage
    features = normalize_features(features)
    #adj = convert_adj_to_sparse(adj)  # this is converted to sparse format due to its immense size
    adj = torch.FloatTensor(adj.to_dense())
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    #print(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(test_idx_reorder)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return g,adj,features,labels,idx_train,idx_val,idx_test



def get_data_v1(cuda=True):

# Here the data is obtained from pytorch-geometric to eliminate unnecessary shuffling done in Kipf's code
    edge_index = pk.load(open("graph.pkl","rb"))
    row,col = edge_index
    edges = [(int(u),int(v)) for u,v in zip(row.tolist(),col.tolist())]
    g = nx.Graph()
    g.add_edges_from(edges)
    adj = np.zeros((torch.max(edge_index).item()+1,torch.max(edge_index).item()+1))
    for u,v in list(g.edges()):
        adj[u,v] = 1
        adj[v,u] = 1    
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])
    adj = sp.sparse.coo_matrix(adj)
    #adj = normalize_adj(adj)
    adj = torch.FloatTensor(adj.todense())
    features = pk.load(open("feature.pkl","rb"))
    features = normalize_features(features.numpy())
    features = torch.FloatTensor(features)
    labels = pk.load(open("label.pkl","rb"))
    idx_train = pk.load(open("train_ids.pkl","rb"))
    idx_val = pk.load(open("valid_ids.pkl","rb"))
    idx_test = pk.load(open("test_ids.pkl","rb"))
    print(len(idx_train),len(idx_test),len(idx_val))
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    return g,adj,features,labels,idx_train,idx_val,idx_test



def get_data_v2(cuda=True):

# Here the data is obtained from pytorch-geometric to eliminate unnecessary shuffling done in Kipf's code
    edge_index = pk.load(open("graph.pkl","rb"))
    row,col = edge_index
    edges = [(int(u),int(v)) for u,v in zip(row.tolist(),col.tolist())]
    g = nx.Graph()
    g.add_edges_from(edges)
    print("Graph Read ")
    adj = np.zeros((torch.max(edge_index).item()+1,torch.max(edge_index).item()+1))
    for u,v in list(g.edges()):
        adj[u,v] = 1
        adj[v,u] = 1    
    adj = nx.to_numpy_array(g,dtype=np.float)
    adj = adj + np.eye(adj.shape[0])
    adj = sp.sparse.coo_matrix(adj)
    print("Adjacency Made")
    adj = torch.FloatTensor(adj.todense())
    features = pk.load(open("feature.pkl","rb"))
    features = normalize_features(features.numpy())
    features = torch.FloatTensor(features)
    print("Features Normalized ")
    labels = pk.load(open("label.pkl","rb"))
    #idx_train = pk.load(open("train_ids.pkl","rb"))
    #idx_val = pk.load(open("valid_ids.pkl","rb"))
    #idx_test = pk.load(open("test_ids.pkl","rb"))
    info = infomap.Infomap("--two-level --silent")
    for e in g1.edges():
        info.addLink(*e)
    info.run()
    c = info.getModules()
    z = defaultdict(list)
    for u in c:
        z[c[u]].append(u)
    c1 = list(z.values())
    c1_arr = np.array(c1)
    #temp_data = np.array([nx.density(g.subgraph(z)) for z in c1],dtype=np.float)
    temp_data = np.array([len(z) for z in c1],dtype=np.float)
    temp_data_idx = np.argsort(temp_data)
    c1_arr = c1_arr[temp_data]
    total = int(0.1 * g.number_of_nodes())
    train_ids = []
    val_ids = []
    test_ids = []
    for z in c1_arr:
        g1 = g.subgraph(z)
        total_node = list(g1,nodes())
        g2 = nx.k_core(g1)
        n1 = list(g2.nodes())
        tar_nodes = np.random.choice(n1,int(0.5*len(n1)),replace=False)
        other_nodes = list(set(total_node).difference(set(tar_nodes)))
        tar_nodes1 = list(set(n1),difference(set(tar_nodes)))
        train_ids.extend(tar_nodes)
        test_ids.extend(other_nodes)
        val_ids.append(tar_nodes1)

    idx_train = np.array(train_ids)
    idx_val = np.array(val_ids)
    idx_test = np.array(test_ids)
    print("Train Validation Test ",len(idx_train),len(idx_val),len(idx_test))
    return g,adj,features,labels,idx_train,idx_val,idx_test
    print("No of Communities {}".format(len(c1)))
         
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    return g,adj,features,labels,idx_train,idx_val,idx_test




def normalize_adj(m):
    print("In adj Nomalizee")
    rowsum = np.array(m.sum(1))
    rowsum = np.power(rowsum,-1).flatten()
    rowsum[np.isinf(rowsum)] = 0  # error handling
    D = sp.sparse.diags(rowsum)
    adj = D.dot(m).dot(D)     # D^-1*A'D^-1
    #print(adj)
    #print(type(adj))
    #adj = np.array(adj,dtype=np.float32)
    return adj


def normalize_features(m):
    rowsum = np.array(m.sum(1))
    z_idx = np.where(rowsum==0)[0]
    rowsum[z_idx] = 1                    # for cases where all features are zeros 
    rowsum = np.power(rowsum,-1).flatten()
    rowsum[np.isinf(rowsum)] = 0  # error handling
    D = sp.sparse.diags(rowsum)
    F = D.dot(m)     # D^-1*F
    return F


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)









#adj,features,labels,idx_train,idx_val,idx_test = get_data()
#print("The Labels")
#print(labels)
