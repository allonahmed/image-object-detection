import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

# loads fashion mnist data directly from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist

# train_images and train_labels are the training set data the model uses to learn
# lables contain the classification options (trouser, shirt, ect...)
# images are 28x28 numpy arrays with pixels ranging from 0 to 255
# model is tested against the test set: the test_images and test_labels arrays
( train_images, train_labels ), ( test_images, test_labels ) = fashion_mnist.load_data()

# we can map our labels to these class names for better data insights
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

### Understanding our dataset:
def ExploreData():
  # we see there are 60000 images and labels in the training set
  print('training image count: ', len(train_images))
  print('training label count: ', len(train_labels))
  # each label is an integer labeled 0 through 9 (this is where we are classifying our images)
  print('train labels:', train_labels)
  # our test set contains 10,000 images represented as 28x28 pixels
  print('test image shape: ', test_images.shape)
  print('test labels: ', test_labels)

ExploreData()

# using matplot to plot x amount of training images and their classification
def PlotImage(count):
  plt.figure(figsize=(10,10))
  for i in range(count):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar(label="colors", orientation="horizontal")
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
  plt.show()

### Building the model

# To build a 'neural network', we must configure the layers of the model, then we compile it
# much of deep learning consists of chaining together these layers.
'''
  Flatten layer method transforms our shape from a 28x28 pixel to a 1d array (28*28 in length). Simply used for formatting our data by lining them up
  Dense layers are then performed in sequence after. The first layer has 128 nodes (or neurons)
  The second dense layer returns a logits array with a length of 10. 
  Each node contains a score that indicates the current image belongs to one of the ten classes
'''
model = tf.keras.Seqential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

