import torch.functional as F
import torch.nn as nn
import torch

device = torch.device("cpu")

############################################################################################################################################
# Dataset
############################################################################################################################################

#
# Last column: age
# Training objective:
# - given other columns, predict age
#

import pandas as pd

data = pd.read_csv("abalone.data", header=None, names=['c' + str(i) for i in range(9)])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = data["c8"].values
data.drop("c8", axis=1, inplace=True)

# Vocabulary encoder 1 2 3
data["label"] =  le.fit_transform(data["c0"].values)
data.drop("c0", axis=1, inplace=True)

x = data.to_numpy()

############################################################################################################################################
# Model
############################################################################################################################################

class Abalone(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(3, 3) # Embedding for gender
        self.fc1 = nn.Linear(10, 10)
        self.act1=nn.ReLU()
        self.fc2 = nn.Linear(10, 20)
        self.act2=nn.ReLU()
        self.solver=nn.Linear(20, 1)

    def forward(self, x):
        x, catx = torch.split(x, 7, dim=1)
        y = self.emb(catx.to(torch.int))[:, 0]
        x = torch.cat([x, y], dim=-1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.solver(x)
        return x

model = Abalone().to(device)

"""
# Torchview bug: force loads the model to cuda
from torchview import draw_graph

model_graph = draw_graph(
	model,
	input_size=(1,8),
	expand_nested=True
)

model_graph.visual_graph.render()
"""

############################################################################################################################################
# Train
############################################################################################################################################

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
loss_fn = l1_loss

Xtr = torch.tensor(x, dtype=torch.float32).to(device)
Ytr = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

n_epochs = 2000
batch_size = 64
patience=50
best_epoch,old_loss=0,1e100

for epoch in range(n_epochs):
	for i in range(0, len(Xtr), batch_size):
		Xbatch = Xtr[i:i+batch_size]
		Ybatch = Ytr[i:i+batch_size]

		Ypreds = model(Xbatch)
		loss = loss_fn(Ypreds, Ybatch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	l1 = l1_loss( model(Xtr), Ytr )
	l2 = l2_loss( model(Xtr), Ytr )

	loss = l1

	print(f"Epoch {epoch}    l1 loss: {l1:.4f}    l2 loss: {l2:.4f}")

	if loss < old_loss:
		best_epoch = epoch
		old_loss = loss
		best_model = model

	if epoch > best_epoch+patience:
		break

import matplotlib.pyplot as plt
Yp = best_model(Xtr).cpu().detach().numpy()
plt.scatter(Ytr.cpu(), Yp)
plt.scatter(Ytr.cpu(), Ytr.cpu())
plt.title("Prediction quality")
plt.xlabel("True age")
plt.ylabel("Predicted age")
plt.show()

embeds = model.emb(Xtr[:, 7].to(int))

from sklearn.decomposition import PCA

trans = PCA(2)
pts = trans.fit_transform(embeds.detach().cpu())

plt.scatter(*pts.T)
plt.title("Gender embedding")
plt.show()
