import pandas as pd
import numpy as np

import torchmetrics
import torch

#
# Homework:
# Try to make it work,
# architecture, layernorm or batchnorm, adam, 1e4 lr
#

device = torch.device("cuda")

####################################################################################
# Dataset
####################################################################################

data = pd.read_csv("chineseMNIST.csv")

Y=data.label.to_numpy()
X=data.drop("label", axis=1).drop("character", axis=1).to_numpy()

print(X.shape,Y.shape)
unique_tags=len(np.unique(Y))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# In [13]: data.label.unique()
# Out[13]:
# array([        9,        10,       100,      1000,     10000, 100000000,
#               0,         1,         2,         3,         4,         5,
#               6,         7,         8])
vocab_y = le.fit(Y)

import torch.nn.functional as F
import torch.nn as nn

Xt = torch.tensor(X, dtype=torch.float32)
Yt=F.one_hot(torch.tensor(vocab_y.transform(Y)).to(int),num_classes=unique_tags).to(float)


####################################################################################
# Model
####################################################################################

class CnClassifier(nn.Module):
	def __init__(self,outClasses):
		super().__init__()
		self.con1=nn.Conv2d(1,20,kernel_size=3,padding=1,stride=2)
		self.act1=nn.ReLU()
		self.con2=nn.Conv2d(20,20,kernel_size=3,padding=1,stride=2)
		self.con3=nn.Conv2d(20,20,kernel_size=3,padding=1,stride=2)
		self.con4=nn.Conv2d(20,20,kernel_size=3,padding=1,stride=2)
		self.con5=nn.Conv2d(20,20,kernel_size=3,padding=1,stride=2)
		self.fl=nn.Flatten()
		self.solver=nn.Linear(80,outClasses)
		self.act_output=nn.Softmax()

	def forward(self, x):
		x = torch.reshape(x, (-1, 1, 64, 64)) # BCWH
		x = self.act1(self.con1(x))
		x = self.act1(self.con2(x))
		x = self.act1(self.con3(x))
		x = self.act1(self.con4(x))
		x = self.act1(self.con5(x))
		x = self.fl(x)

		x = self.solver(x)
		return x

	def call(self, x):
		x = self.act_output(self.forward(x))
		return x

model = CnClassifier(unique_tags).to(device)
f1_metric = torchmetrics.F1Score("multiclass", num_classes=15).to(device)

Xt = Xt.to(device)
Yt = Yt.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

batch_size=64
max_epochs=1000

x_batches = torch.split(Xt, batch_size)
y_batches = torch.split(Yt, batch_size)

for epoch in range(max_epochs):
	for bx, by in zip(x_batches, y_batches):
		bp = model(bx)

		loss = loss_fn(bp, by)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	with torch.no_grad():
		preds = torch.cat([torch.argmax(model(minibatch), 1) for minibatch in x_batches])
		truth = torch.argmax(Yt, 1)

		acc = torch.mean( 1.0*(preds == truth) )
		f1 = f1_metric(preds, truth)

		print(f'Finished epoch {epoch}, latest loss {loss:.4f}, F1 {f1:.2f}, accuracy {acc:.2f}')

"""
for epoch in range(n_epochs):
    QM, QMn = 0,0
    for i in range(0, len(Xt), batch_size):
        Xbatch = Xt[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = Yt[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            QM += metric(ybatch, y_pred)
            QMn += 1

    print(f'Finished epoch {epoch}, latest loss {loss}, QM {QM/QMn}')
"""


####################################################################################
# Evals
####################################################################################

from sklearn.metrics import classification_report

truth = torch.argmax(Yt, 1).cpu()
preds = torch.cat([torch.argmax(model(minibatch), 1) for minibatch in torch.split(Xt, batch_size)]).cpu()

report = classification_report(truth, preds)

print(report)
