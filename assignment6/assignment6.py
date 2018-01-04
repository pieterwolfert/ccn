import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from utils import get_mnist

train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=False, classes = [0])

class Generator(chainer.Chain):
   def __init__(self, n_hidden, bottom_width=3, ch=512):
       super(Generator, self).__init__()
       self.n_hidden = n_hidden
       self.ch = ch
       self.bottom_width = bottom_width
       
       with self.init_scope():
           self.l1 = self.Linear(None, n_hidden)
           self.bn1 = L.BatchNormalization(size=ch)
           self.deconv = self.Deconvolution2D(in_channels=ch, out_channels=1, outsize=(28,28))
   def __call__(self, z):
      h = F.relu(self.bn1(self.l1(z)))
      x = F.sigmoid(self.deconv(h))
      return x

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=6, ksize=5, stride=1)
            self.conv2 = L.Convolution2D(
                in_channels=6, out_channels=16, ksize=5, stride=1)
            self.conv3 = L.Convolution2D(
                in_channels=16, out_channels=120, ksize=4, stride=1)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, 1)
            
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv3(h))
        h = F.sigmoid(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))