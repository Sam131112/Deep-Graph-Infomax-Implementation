import torch

class Discriminator(torch.nn.Module):
    def __init__(self,f_dim1,f_dim2):
        super(Discriminator,self).__init__()
        self.disc = torch.nn.Bilinear(f_dim2,f_dim2,1)
        torch.nn.init.xavier_uniform_(self.disc.weight.data)
        self.disc.bias.data.fill_(0)


    def forward(self,feats,c):

        summary = c.repeat((feats.size()[0],1))
        d1 = self.disc(feats,summary)
        return d1






