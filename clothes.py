import tensorflow as tf
import numpy
import matplotlib.pyplot as pytplot

# loads fashion mnist data directly from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist

# train_images and train_labels are the training set data the model uses to learn
# model is tested agaijnmst the test set: the test_images and test_labels arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

