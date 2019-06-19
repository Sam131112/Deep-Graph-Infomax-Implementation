import torch
import torch.nn as nn
import torch.nn.functional as F
from my_layer import GraphConvolutionLayer
from my_layer import GraphSageLayer
import math


class SimpleModel(nn.Module):
    def __init__(self,nfeat,nhid,nclass):
        super(SimpleModel,self).__init__()
        self.fc1 = nn.Linear(nfeat,nhid)
        self.fc2 = nn.Linear(nhid,nhid)
        self.fc3 = nn.Linear(nhid,nclass)
        self.drop = nn.Dropout(0.5)
        torch.nn.init.xavier_normal_(self.fc1.weight.data)
        torch.nn.init.normal_(self.fc1.bias.data)
        torch.nn.init.xavier_normal_(self.fc2.weight.data)
        torch.nn.init.normal_(self.fc2.bias.data)
        torch.nn.init.xavier_normal_(self.fc3.weight.data)
        torch.nn.init.normal_(self.fc3.bias.data)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

class GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN,self).__init__()
        #self.gcn1 = GraphConvolutionLayer(nfeat,nhid)
        #self.gcn2 = GraphConvolutionLayer(nhid,nhid)
        self.gcn1 = GraphSageLayer(nfeat,nhid)
        self.gcn2 = GraphSageLayer(nhid,nhid)
        self.fc1 = nn.Linear(nhid,nclass)
        self.glorot(self.fc1.weight.data)    # Glorot Inialization
        self.zeros(self.fc1.bias.data)
        #torch.nn.init.xavier_normal_(self.fc1.weight.data)
        #torch.nn.init.normal_(self.fc1.bias.data)
        self.dropout = dropout

    def glorot(self,tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0/(tensor.size(-2)+tensor.size(-1)))
            tensor.data.uniform_(-stdv,stdv)
    def zeros(self,tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def forward(self,x,adj):
        x = F.relu(self.gcn1(x,adj))
        #print("Train ",x)
        x = F.dropout(x,self.dropout)
        x = self.gcn2(x,adj)
        x = self.fc1(x)
        #print("Train ",x)
        return F.log_softmax(x,dim=1)



