import torch
from readout import Readout
from discriminator import Discriminator
from my_layer import GraphConvolutionLayer


class DeepGraphInfomax(torch.nn.Module):
    def __init__(self,feats,c_feats,out_feats):
        super(DeepGraphInfomax,self).__init__()
        self.read = Readout()
        self.disc = Discriminator(out_feats,out_feats)
        self.gcn = GraphConvolutionLayer(feats,out_feats)


    def forward(self,feats,c_feats,adj):

        feats1 = self.gcn(feats,adj)
        c = self.read(feats1)
        c = c.view(1,-1)
        h1 = self.disc(feats1,c)
        feats2 = self.gcn(c_feats,adj)
        h2 = self.disc(feats2,c)
        h = torch.cat((h1,h2),0)
        return h

    def embeddings(self,feature,adj):
        embed= self.gcn(feature,adj)
        return embed






















