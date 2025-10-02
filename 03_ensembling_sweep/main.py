from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# pip install paxplot
import paxplot

x,y=make_blobs(n_samples=10000,centers=7,cluster_std=4,n_features=2)

def plot_dataset():
	plt.scatter(*x.T, c=y)
	plt.show()

#
# 1/9 Bagging trees
#
def decision_tree_classifier_bagging():

	# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
	p_grid = {
		"estimator": [
			DecisionTreeClassifier(max_depth=1),
			DecisionTreeClassifier(max_depth=2),
			DecisionTreeClassifier(max_depth=3),
			DecisionTreeClassifier(max_depth=4),
			DecisionTreeClassifier(max_depth=5),
		],
		"n_estimators":[2, 3, 4, 5]
	}

	clf = GridSearchCV(BaggingClassifier(), p_grid)
	clf.fit(x, y)

	scores = clf.cv_results_.get("mean_test_score")
	params = clf.cv_results_.get("params")

	grid = []

	for score, param in zip(scores, params):
		a = param["estimator"].max_depth
		b = param["n_estimators"]

		grid.append([a, b, score])

	paxfig = paxplot.pax_parallel(n_axes=3)
	paxfig.plot(grid)

	paxfig.add_colorbar(
		ax_idx=2,
		cmap="viridis",
		colorbar_kwargs={"label": "Score"}
	)

	paxfig.set_labels(["max_depth", "n_estimators", "score"])

	plt.suptitle("BaggingClassifier(DecisionTreeClassifier(...), ...)")
	plt.show()

#
# 2/9 Stacking trees
#
def decision_tree_classifier_stacking():
	estimators = []

	for i in range(1, 6):
		for j in range(1, 6):
			estimators.append( [ (f"model{m}", DecisionTreeClassifier(max_depth=i)) for m in range(j) ] )

	# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
	p_grid = {
		"estimators": estimators
	}

	clf = GridSearchCV(StackingClassifier([]), p_grid)
	clf.fit(x, y)

	scores = clf.cv_results_.get("mean_test_score")
	params = clf.cv_results_.get("params")

	grid = []

	for score, param in zip(scores, params):
		a = param["estimators"][0][1].max_depth
		b = len(param["estimators"])

		grid.append([a, b, score])

	paxfig = paxplot.pax_parallel(n_axes=3)
	paxfig.plot(grid)

	paxfig.add_colorbar(
		ax_idx=2,
		cmap="viridis",
		colorbar_kwargs={"label": "Score"}
	)

	paxfig.set_labels(["max_depth", "n_estimators", "score"])

	plt.suptitle("StackingClassifier([ ('', DecisionTreeClassifier(...)), ...])")
	plt.show()

#
# 3/9 Boosting trees
#
def decision_tree_classifier_boosting():

	p_grid = {
		"estimator": [
			DecisionTreeClassifier(max_depth=1),
			DecisionTreeClassifier(max_depth=2),
			DecisionTreeClassifier(max_depth=3),
			DecisionTreeClassifier(max_depth=4),
			DecisionTreeClassifier(max_depth=5),
		],
		"n_estimators":[2, 3, 4, 5]
	}

	clf = GridSearchCV(AdaBoostClassifier(), p_grid)
	clf.fit(x, y)

	scores = clf.cv_results_.get("mean_test_score")
	params = clf.cv_results_.get("params")

	grid = []

	for score, param in zip(scores, params):
		a = param["estimator"].max_depth
		b = param["n_estimators"]

		grid.append([a, b, score])

	paxfig = paxplot.pax_parallel(n_axes=3)
	paxfig.plot(grid)

	paxfig.add_colorbar(
		ax_idx=2,
		cmap="viridis",
		colorbar_kwargs={"label": "Score"}
	)

	paxfig.set_labels(["max_depth", "n_estimators", "score"])

	plt.suptitle("AdaBoostClassifier(DecisionTreeClassifier(...), ...)")
	plt.show()

#
# 4/9 Bagging MLPs
#
def mlp_bagging():
	estimator = []

	for i in range(1, 6):
		for j in range(1, 6):
			estimator.append( MLPClassifier( (i, j), max_iter=10 ) )

	p_grid = {
		"estimator": estimator,
		"n_estimators":[2, 3] # 4, 5 = Too long
	}

	clf = GridSearchCV(BaggingClassifier(), p_grid, n_jobs=-1)
	clf.fit(x, y)

	scores = clf.cv_results_.get("mean_test_score")
	params = clf.cv_results_.get("params")

	grid = []

	for score, param in zip(scores, params):
		a = param["n_estimators"]
		b, c = param["estimator"].hidden_layer_sizes

		grid.append([a, b, c, score])

	paxfig = paxplot.pax_parallel(n_axes=4)
	paxfig.plot(grid)

	paxfig.add_colorbar(
		ax_idx=3,
		cmap="viridis",
		colorbar_kwargs={"label": "Score"}
	)

	paxfig.set_labels(["max_depth", "layer.1", "layer.2", "score"])

	plt.suptitle("BaggingClassifier(MLPClassifier(...), ...)")
	plt.show()

#
# 5/9 Stacking MLPs
#
def mlp_stacking():
	estimators = []

	for m in range(2, 4):
		for i in range(1, 6):
			for j in range(1, 6):
				estimators.append( [ (f"model{k}", MLPClassifier( (i, j), max_iter=10 ) ) for k in range(m) ] )

	p_grid = {
		"estimators": estimators,
	}

	clf = GridSearchCV(StackingClassifier([]), p_grid, n_jobs=-1)
	clf.fit(x, y)

	scores = clf.cv_results_.get("mean_test_score")
	params = clf.cv_results_.get("params")

	grid = []

	for score, param in zip(scores, params):
		a = len(param["estimators"])
		b, c = param["estimators"][0][1].hidden_layer_sizes

		grid.append([a, b, c, score])

	paxfig = paxplot.pax_parallel(n_axes=4)
	paxfig.plot(grid)

	paxfig.add_colorbar(
		ax_idx=3,
		cmap="viridis",
		colorbar_kwargs={"label": "Score"}
	)

	paxfig.set_labels(["max_depth", "layer.1", "layer.2", "score"])

	plt.suptitle("StackingClassifier(MLPClassifier(...), ...)")
	plt.show()

#
# 6/9 Boosting MLPs
#
def mlp_boosting():
	estimator = []

	# Poor numeric stability?
	for i in range(4, 6):
		for j in range(4, 6):
			estimator.append( MLPClassifier( (i, j), max_iter=10 ) )

	p_grid = {
		"estimator": estimator,
		"n_estimators":[2, 3] # 4, 5 = Too long
	}

	clf = GridSearchCV(AdaBoostClassifier(), p_grid, n_jobs=-1)
	clf.fit(x, y)

	scores = clf.cv_results_.get("mean_test_score")
	params = clf.cv_results_.get("params")

	grid = []

	for score, param in zip(scores, params):
		a = param["n_estimators"]
		b, c = param["estimator"].hidden_layer_sizes

		grid.append([a, b, c, score])

	paxfig = paxplot.pax_parallel(n_axes=4)
	paxfig.plot(grid)

	paxfig.add_colorbar(
		ax_idx=3,
		cmap="viridis",
		colorbar_kwargs={"label": "Score"}
	)

	paxfig.set_labels(["max_depth", "layer.1", "layer.2", "score"])

	plt.suptitle("StackingClassifier(MLPClassifier(...), ...)")
	plt.show()

mlp_boosting()

def svc():

	p_grid = {
		"kernel": ["linear"],
		"C":[1, 10]
	}

	svc = SVC()

	clf = GridSearchCV(svc, p_grid)


