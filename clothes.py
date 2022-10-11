import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

# loads fashion mnist data directly from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist

# train_images and train_labels are the training set data the model uses to learn
# lables contain the classification options (trouser, shirt, ect...)
# images are 28x28 numpy arrays with pixels ranging from 0 to 255
# model is tested against the test set: the test_images and test_labels arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

def PlotImage():
  plt.figure()
  plt.imshow(train_images[0])
  plt.colorbar()
  plt.grid(False)
  plt.show()

PlotImage()