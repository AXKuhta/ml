import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data = fetch_olivetti_faces()
x = data.get("data")
y = data.get("target")

y_ = label_binarize(y, classes=np.arange(40))

# plt.plot(recall[i], precision[i], lw=2)
# plt.bar(np.arange(40), aucrp)
# plt.show()
class results:
	class straight_forest:
		precision = [None]*40
		recall = [None]*40
		aucrp = [None]*40

	class pca_forest:
		precision = [None]*40
		recall = [None]*40
		aucrp = [None]*40

	class tsne_forest:
		precision = [None]*40
		recall = [None]*40
		aucrp = [None]*40

	class straight_mlp:
		precision = [None]*40
		recall = [None]*40
		aucrp = [None]*40

	class pca_mlp:
		precision = [None]*40
		recall = [None]*40
		aucrp = [None]*40

	class tsne_mlp:
		precision = [None]*40
		recall = [None]*40
		aucrp = [None]*40

# Small dataset so avoid splitting
# x_train, x_test, y_train, y_test = train_test_split(x, y)

#
# 1. No feature extraction, random forest
#
clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(x, y)

# https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier#56092736
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#
# thresholds too narrow =  recall <<< precision
# thresholds too wide =    recall >>> precision
#
#
probs = clf.predict_proba(x)

for i in range(40):
	precision, recall, _ = precision_recall_curve(y == i, probs[:, i])
	aucrp = auc(recall, precision)

	results.straight_forest.precision[i] = precision
	results.straight_forest.recall[i] = recall
	results.straight_forest.aucrp[i] = aucrp

print(clf.score(x, y))

#
# 2. PCA, random forest
#
trans = PCA(7, random_state=42)
x_pca = trans.fit_transform(x)

clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(x_pca, y)

probs = clf.predict_proba(x_pca)

for i in range(40):
	precision, recall, _ = precision_recall_curve(y == i, probs[:, i])
	aucrp = auc(recall, precision)

	results.pca_forest.precision[i] = precision
	results.pca_forest.recall[i] = recall
	results.pca_forest.aucrp[i] = aucrp

print(clf.score(x_pca, y))

#
# 3. TSNE, random forest
#
trans = TSNE(7, method="exact", random_state=42) # Eat the n^2 time
x_tsne = trans.fit_transform(x)

clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(x_tsne, y)

probs = clf.predict_proba(x_tsne)

for i in range(40):
	precision, recall, _ = precision_recall_curve(y == i, probs[:, i])
	aucrp = auc(recall, precision)

	results.tsne_forest.precision[i] = precision
	results.tsne_forest.recall[i] = recall
	results.tsne_forest.aucrp[i] = aucrp

print(clf.score(x_tsne, y))

#
# 4. No feature extraction, MLP
#
# parameters >>> samples lmao
#
trans = StandardScaler() # critically important
x_norm = trans.fit_transform(x)

#clf = MLPClassifier(hidden_layer_sizes=[10], batch_size=80, learning_rate="adaptive", solver="sgd", early_stopping=True, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=[10], batch_size=80, solver="adam", early_stopping=True, random_state=42)
clf.fit(x_norm, y)

probs = clf.predict_proba(x_norm)

for i in range(40):
	precision, recall, _ = precision_recall_curve(y == i, probs[:, i])
	aucrp = auc(recall, precision)

	results.straight_mlp.precision[i] = precision
	results.straight_mlp.recall[i] = recall
	results.straight_mlp.aucrp[i] = aucrp

print(clf.score(x_norm, y))

#
# 5. PCA, MLP
#
trans = PCA(7, random_state=42)
x_pca = trans.fit_transform(x)

trans = StandardScaler()
x_pca_norm = trans.fit_transform(x_pca)

#clf = MLPClassifier(hidden_layer_sizes=[10], batch_size=80, learning_rate="adaptive", solver="sgd", early_stopping=True, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=[10], batch_size=80, solver="adam", early_stopping=True, random_state=42)
clf.fit(x_pca_norm, y)

probs = clf.predict_proba(x_pca_norm)

for i in range(40):
	precision, recall, _ = precision_recall_curve(y == i, probs[:, i])
	aucrp = auc(recall, precision)

	results.pca_mlp.precision[i] = precision
	results.pca_mlp.recall[i] = recall
	results.pca_mlp.aucrp[i] = aucrp

print(clf.score(x_pca_norm, y))

#
# 6. TSNE, MLP
#
trans = TSNE(7, method="exact", random_state=42)
x_tsne = trans.fit_transform(x)

trans = StandardScaler()
x_tsne_norm = trans.fit_transform(x_tsne)

#clf = MLPClassifier(hidden_layer_sizes=[10], batch_size=80, learning_rate="adaptive", solver="sgd", early_stopping=True, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=[10], batch_size=80, solver="adam", early_stopping=True, random_state=42)
clf.fit(x_tsne_norm, y)

probs = clf.predict_proba(x_tsne_norm)

for i in range(40):
	precision, recall, _ = precision_recall_curve(y == i, probs[:, i])
	aucrp = auc(recall, precision)

	results.tsne_mlp.precision[i] = precision
	results.tsne_mlp.recall[i] = recall
	results.tsne_mlp.aucrp[i] = aucrp

print(clf.score(x_tsne_norm, y))


#
# Final curves
#
import seaborn as sns

# Six bars, 40 hues

hue = np.arange(40).tolist()
hues = []

x = ["forest"]*40 + ["pca forest"]*40 + ["tsne forest"]*40 + ["mlp"]*40 + ["pca mlp"]*40 + ["tsne mlp"]*40
y = results.straight_forest.aucrp + \
results.pca_forest.aucrp + \
results.tsne_forest.aucrp + \
results.straight_mlp.aucrp + \
results.pca_mlp.aucrp + \
results.tsne_mlp.aucrp
hues = hue + hue + hue + hue + hue + hue

sns.barplot(x=x, y=y, hue=hues)
plt.title("aucrp for 40 classes on 6 classifiers")
plt.ylabel("aucrp")
plt.show()
