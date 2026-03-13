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

data = pd.read_csv("chineseMNIST.csv.gz")

y = data.label.to_numpy()
x = data.drop("label", axis=1).drop("character", axis=1).to_numpy()

print(x.shape,y.shape)
unique_tags=len(np.unique(y))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# In [13]: data.label.unique()
# Out[13]:
# array([        9,        10,       100,      1000,     10000, 100000000,
#               0,         1,         2,         3,         4,         5,
#               6,         7,         8])
vocab_y = le.fit(y)

train_x_, test_x_, train_y_, test_y_ = train_test_split(x, y, random_state=42)

import torch.nn.functional as F
import torch.nn as nn

train_x = torch.tensor(train_x_, dtype=torch.float32)
train_y = F.one_hot(torch.tensor(vocab_y.transform(train_y_)).to(int), num_classes=unique_tags).to(float)

test_x = torch.tensor(test_x_, dtype=torch.float32)
test_y = F.one_hot(torch.tensor(vocab_y.transform(test_y_)).to(int), num_classes=unique_tags).to(float)


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
		self.att=nn.TransformerEncoderLayer(80, 1, 80) # 3x Dense layers + Attention
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

		x = self.att(x)
		x = self.solver(x)
		return x

	def call(self, x):
		x = self.act_output(self.forward(x))
		return x

model = CnClassifier(unique_tags).to(device)
f1_metric = torchmetrics.F1Score("multiclass", num_classes=15).to(device)
ap_metric = torchmetrics.AveragePrecision("multiclass", num_classes=15).to(device)

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

batch_size=64
max_epochs=100

x_batches = torch.split(train_x, batch_size)
y_batches = torch.split(train_y, batch_size)

for epoch in range(max_epochs):
	for bx, by in zip(x_batches, y_batches):
		bp = model(bx)

		loss = loss_fn(bp, by)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	with torch.no_grad():
		probs = [model(minibatch) for minibatch in x_batches]
		preds = torch.cat([torch.argmax(x, 1) for x in probs])
		truth = torch.argmax(train_y, 1)

		acc = torch.mean( 1.0*(preds == truth) )
		f1 = f1_metric(preds, truth)
		ap = ap_metric(torch.vstack(probs), truth)

		print(f'Finished epoch {epoch}, latest loss {loss:.4f}, F1 {f1:.2f}, AP {ap:.2f}, accuracy {acc:.2f}')

####################################################################################
# Evals
####################################################################################

from sklearn.metrics import classification_report

truth = torch.argmax(test_y, 1).cpu()
preds = torch.cat([torch.argmax(model(minibatch), 1) for minibatch in torch.split(test_x, batch_size)]).cpu()

report = classification_report(truth, preds)

print("Results on test:")
print(report)
