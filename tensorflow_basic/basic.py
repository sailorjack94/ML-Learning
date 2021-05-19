# Into to TensorFlow

import tensorflow as tf 
from tensorflow import keras


# Two sets of data - training and testing. Each has a group of images, and a group of labels in number format. Using numbers to avoid bias.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    # First layer has same size as image diameters
    keras.layers.Flatten(input_shape=(28,28)),
    # 128 function with parameters. Aim is that the output of all functions together equals one of the 10 values below. Logic must figure out the contents of each of these functions.
    # RELU = rectified linear unit. Returns a value if result is greater than 0.
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Returns one of 10 values
    # Approximates probability detection to speed up computation
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# Build model with opt/loss functions
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy')

# Loading training data
model.fit(train_images, train_labels, epochs = 5)

# Model hasn't seen test images, evaluate method determines efficacy of the model.
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Execture model on our own dataset.
# predictions = model.predict(my_images)
