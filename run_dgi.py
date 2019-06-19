from deepgraphinfomax import DeepGraphInfomax
from discriminator import Discriminator
from my_utils import get_data_v1,accuracy
from my_model import SimpleModel
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.cuda.set_device(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g,adj,features,labels,idx_train,idx_val,idx_test = get_data_v1(True)

idx_perm = np.random.permutation(np.arange(features.shape[0]))
c_features = features[idx_perm]

num_feats = features.shape[1]

model =  DeepGraphInfomax(num_feats,num_feats,128)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=0.0)
loss = torch.nn.BCEWithLogitsLoss()
or_gold = torch.ones((adj.shape[0],1))
cr_gold = torch.zeros((adj.shape[0],1))

gold = torch.cat((or_gold,cr_gold),dim=0)
gold = gold.cuda()

z = model(features,c_features,adj)
ls = loss(z,gold)
print(z.shape,gold.shape)
print(ls)

h = model.embeddings(features,adj)
print(h.shape)


for _ in range(200):
    
    idx_perm = np.random.permutation(np.arange(features.shape[0]))
    c_features = features[idx_perm]
    optimizer.zero_grad()
    y_hat = model(features,c_features,adj)
    output = loss(y_hat,gold)
    output.backward()
    optimizer.step()



embed = model.embeddings(features,adj)

embed= embed.cpu().detach()
idx_train = idx_train.cpu().detach()
idx_test = idx_test.cpu().detach()
labels = labels.cpu().detach()


train_feats = embed[idx_train]

num_class = torch.max(labels)+1

Model = SimpleModel(128,64,torch.max(labels).item()+1)


Loss = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.01,weight_decay=0.0)

for _ in range(200):
    optimizer.zero_grad()
    y_hat = Model(embed)
    output = Loss(y_hat[idx_train],labels[idx_train])
    output.backward()
    optimizer.step()


y_hat = Model(embed[idx_test])
lbl = labels[idx_test]

print(accuracy(y_hat,lbl).item())

