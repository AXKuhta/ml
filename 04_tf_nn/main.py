import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("circles.dat")

x = data[:, :2]
y = data[:, 2]

plt.scatter(*x.T, c=y)
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model

# Wider net
# Accuracy 0.84
model = Sequential([
	Input(shape=(2,)),
	Dense(150, activation="relu"),
	Dense(150, activation="relu"),
	Dense(1, activation="sigmoid")
])

model.compile(
	optimizer="adam",
	loss="binary_crossentropy",
	metrics=["accuracy"]
)

plot_model(model, show_shapes=True, show_layer_activations=True)
plt.show()

# Batch size = dataset size
model.fit(x, y, batch_size=6700, epochs=10000)

yp = model.predict(x)

clr = plt.scatter(*x.T, c=yp)
plt.colorbar(clr)
plt.show()

from mlxtend.plotting import plot_decision_regions

yp = (yp>0.5).astype(int)

plot_decision_regions(x, yp[:, 0], clf=model, legend=2)
plt.show()
