import torch
import math
from torch.nn import Parameter


class GraphConvolutionLayer(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolutionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = torch.nn.Linear(in_features,out_features)
        self.glorot(self.fc1.weight.data)
        self.zeros(self.fc1.bias.data)
        #torch.nn.init.xavier_normal_(self.fc1.weight.data)
        #torch.nn.init.normal_(self.fc1.bias.data)

    def glorot(self,tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0/(tensor.size(-2)+tensor.size(-1)))
            tensor.data.uniform_(-stdv,stdv)
    def zeros(self,tensor):
        if tensor is not None:
            tensor.data.fill_(0)



    def forward(self,input,adj):
        support = torch.mm(adj,input)
        output = self.fc1(support)
        return output



class GraphSageLayer(torch.nn.Module):
    def __init__(self,in_feature,out_feature,bias=True):
        super(GraphSageLayer,self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(torch.Tensor(2*in_feature, out_feature))
        if bias:
            self.bias = Parameter(torch.Tensor(out_feature))

        self.glorot(self.weight)
        self.zeros(self.bias)

    def glorot(self,tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0/(tensor.size(-2)+tensor.size(-1)))
            tensor.data.uniform_(-stdv,stdv)
    def zeros(self,tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    
    def forward(self,x,adj):
        #_x = torch.empty((x.size(0),x.size(0),x.size(1)))
        x1 = torch.empty((x.size(0),x.size(1))).cuda()
        #x1 = torch.empty((x.size(0),x.size(1)))
        #print("Out for loop",x)
        for j in range(x.size(0)):
                tmp = adj[j] == 1
                temp = x[tmp]
                #print("For loop ",j,temp)
                temp = torch.mean(temp,0)
                x1[j] = temp
                #print("For loop ",j,temp)
        #print("aggregate ",x1)
        n_x = torch.cat((x1,x),1)
        #print("New Shape ",n_x.shape)
        n_x = torch.matmul(n_x,self.weight)
        #print("New Shape ",n_x.shape)
        n_x = n_x+self.bias
        return n_x



class CoreSageLayer(torch.nn.Module):
    def __init__(self,in_feature,out_feature,bias=True):
        super(GraphSageLayer,self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = Parameter(torch.Tensor(3,2*in_feature, out_feature))
        if bias:
            self.bias = Parameter(torch.Tensor(out_feature))

        self.glorot(self.weight)
        self.zeros(self.bias)

    def glorot(self,tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0/(tensor.size(-2)+tensor.size(-1)))
            tensor.data.uniform_(-stdv,stdv)
    def zeros(self,tensor):
        if tensor is not None:
            tensor.data.fill_(0)


    def forward(self,g,x,adj):
        #_x = torch.empty((x.size(0),x.size(0),x.size(1)))
        x1 = torch.empty((x.size(0),x.size(1))).cuda()
        #x1 = torch.empty((x.size(0),x.size(1)))
        #print("Out for loop",x)
        for j in range(x.size(0)):
                tmp = adj[j] == 1
                temp = x[tmp]
                #print("For loop ",j,temp)
                temp = torch.mean(temp,0)
                x1[j] = temp
                #print("For loop ",j,temp)
        #print("aggregate ",x1)
        n_x = torch.cat((x1,x),1)
        #print("New Shape ",n_x.shape)
        n_x = torch.matmul(n_x,self.weight)
        #print("New Shape ",n_x.shape)
        n_x = n_x+self.bias
        return n_x
