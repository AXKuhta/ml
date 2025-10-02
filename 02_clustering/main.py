from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

n_samples = 1500
random_state = 170

transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

#
# Datasets
#
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation) # Anisotropic blobs
X_varied, y_varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_filtered = np.hstack((y[y == 0][:500], y[y == 1][:100], y[y == 2][:10]))

datasets = [
	(X, y),
	(X_aniso, y),
	(X_varied, y_varied),
	(X_filtered, y_filtered)
]

def show_datasets():
	for X, y in datasets:
		plt.scatter(*X.T, c=y)
		plt.show()

#
# Model
#
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

#
# Gaussian mixture
# + hparam tuning
#
for k, (X, y) in enumerate(datasets):
	rand_score_log = []
	bic_log = []
	c_log = []
	plt_x = []

	for i in range(2, 10):
		clf = GaussianMixture(n_components=i)
		clf.fit(X)

		c = clf.predict(X)

		#
		# Metrics
		#
		rand_score = adjusted_rand_score(y, c)
		bic = clf.bic(X)

		rand_score_log.append(rand_score)
		bic_log.append(bic)
		plt_x.append(i)
		c_log.append(c)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10, 4])
	ax1.plot(plt_x, rand_score_log, marker="o")
	ax2.plot(plt_x, bic_log, marker="o")
	ax3.scatter(*X.T, c=c_log[1])
	ax1.set_title("Adjusted rand score")
	ax2.set_title("BIC (low is good)")
	ax3.set_title("Result (3 clusters)")
	fig.show()

#
# K-Means
# + hparam tuning
#
# https://scikit-learn.org/stable/modules/clustering.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#
for k, (X, y) in enumerate(datasets):
	rand_score_log = []
	silhouette_log = []
	c_log = []
	plt_x = []

	for i in range(2, 10):
		clf = KMeans(n_clusters=i)
		clf.fit(X)

		c = clf.predict(X)

		#
		# Metrics
		#
		rand_score = adjusted_rand_score(y, c)
		silhouette = silhouette_score(X, c)

		rand_score_log.append(rand_score)
		silhouette_log.append(silhouette)
		plt_x.append(i)
		c_log.append(c)

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10, 4])
	ax1.plot(plt_x, rand_score_log, marker="o")
	ax2.plot(plt_x, silhouette_log, marker="o")
	ax3.scatter(*X.T, c=c_log[1])
	ax1.set_title("Adjusted rand score")
	ax2.set_title("Silhouette score")
	ax3.set_title("Result (3 clusters)")
	fig.show()
