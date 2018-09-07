from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def plot_data(pl, X, y):
    pl.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    pl.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl

# Common function that draws the decision boundaries
def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1

    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make prediction with the model and reshape the output so contourf can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    # plot the contour
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    # plot the moons of data
    plot_data(plt, X, y)

    return plt

"""
Generate some data blobs. Data will be either 0 or 1 when 2 is number of centers.
X is a [number of samples, 2] sized array. X[sample] contains its x,y position of the sample in the sapce
ex: X[1] = [1.342, -2.3], X[2] = [-4.342, 2.12]
y is a [number of samples] sized array. y[sample] contains the class index (ie. 0 or 1 when there are 2 centers)
ex: y[1] = 0, y[1] = 1
"""
X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)

# pl = plot_data(plt, X, y)
# pl.show()

# Split the data into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the keras model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Simple Sequential Model - 
# Each layer is inserted at the end of the network and gets the input from 
#   the previous layer or the data passed in, in the case of the first layer.
model = Sequential()

# Train to divide the 2 classes.  1 neuron: class 1 or 2
model.add(Dense(4, input_shape=(2,), activation="tanh", name="Hidden-1"))
model.add(Dense(4, activation="tanh", name="Hidden-2"))
model.add(Dense(1, input_shape=(2,), activation="sigmoid", name="Output_layer"))
model.summary()

# 'binary_crossentropy' used to calculate loss
# metrics = what we want to optimize
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
from keras.utils import plot_model
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

# Define early stopping callback
from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor='val_acc', patience=5, mode=max)]
# 100 runs through the training data, on each run we update the accuracy
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))

# Get loss and accuracy on test data
eval_result = model.evaluate(X_test, y_test)

# Print test accuracy
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])

plot_decision_boundary(model, X, y).show()
