from rbm import RBM
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#Loading in the mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#Img size
RBM_visible_sizes = 784
RBM_hidden_sizes = 600

#build and test a model with MNIST data
rbm = RBM(RBM_visible_sizes, RBM_hidden_sizes,verbose=1)
rbm.fit(trX,teX)
