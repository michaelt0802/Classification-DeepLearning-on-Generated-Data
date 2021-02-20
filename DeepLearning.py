import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_moons
from pylab import rcParams
from tensorflow import keras
import warnings  
warnings.filterwarnings('ignore')

# Create the training data
np.random.seed(42) 
data, labels = make_moons(n_samples=500, noise=0.1)
colors = ['r' if y else 'b' for y in labels]
print('data.shape =', data.shape)
print('labels.shape =', labels.shape)
plt.scatter(data[:,0], data[:,1], c=colors)
plt.show()

# Create the test data
np.random.seed(17)   
test_data, test_labels = make_moons(n_samples=500, noise=0.1)
colors = ['r' if y else 'b' for y in test_labels]
print('test_data.shape =', test_data.shape)
print('test_labels.shape =', test_labels.shape)
plt.scatter(test_data[:,0], test_data[:,1], c=colors)
plt.show()


Dense = keras.layers.Dense
Sequential = keras.Sequential

model = Sequential()

model.add(Dense(15, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(5, activation='softmax'))

# The output layer is a single neuron with the sigmoid activation function.
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, 
                    validation_split = 0.2, 
                    epochs = 300, 
                    batch_size = 50)

# Evaluate the model's performance
train_loss, train_acc = model.evaluate(data, labels)
test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Training set accuracy:', train_acc)
print('Test set accuracy:', test_acc)

model.summary()

# The history of our accuracy during training.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# The history of our cross-entropy loss during training.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#
# This code is substantively from https://rohitmidha23.github.io/Neural-Network-Decision-Boundary/
#
def plot_decision_boundary(X, y, model, steps=1000, cmap='bwr'):
    # The following allows you to adjust the plot size
    rcParams['figure.figsize'] = 8, 6  # 8 inches by 6 inches
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)

    return fig, ax

plot_decision_boundary(test_data, test_labels, model) 
# Reset figure size back to default.
rcParams['figure.figsize'] = 6, 4