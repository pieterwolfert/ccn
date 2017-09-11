# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from utils import RandomIterator, get_mnist
import matplotlib.pyplot as plt
import matplotlib

class MLP(Chain):
    """
    
    Implementation of a 3-layer Multilayer Perceptron
    
    """
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        
        self.l1 = L.Linear(None, n_units) # Input layer n_units -> n_units
        self.l2 = L.Linear(None, n_units)
        self.l3 = L.Linear(n_units, n_out)
        
    def __call__(self, x):
        h1 = self.l1(x)
        h2 = F.relu(self.l2(h1))
        output = F.relu(self.l3(h2))
        return output
  
class Classifier(Chain):
    """
    Classifier chain on top of MLP chain 
    for accuracy evaluation and prediction
    
    """
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
    
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

# Retrieve train & test data      
train, test = get_mnist()
# split test inputs and labels
inputs, labels = np.array([tup[0] for tup in test]), np.array([tup[1] for tup in test])
# Set up model & classifier
model = MLP(10,10)
classifier = Classifier(model)
optimizer = optimizers.SGD()
optimizer.setup(classifier)


iterator = RandomIterator(train,32)

av_loss = []
ep_loss = []
test_loss = []
for epoch in range(1,21):
     for batch in iterator:
        sources,targets = batch[0], batch[1]
        classifier.cleargrads()
        loss = F.sum(classifier(sources, targets))
        loss.backward()
        av_loss.append(loss.data)    
        optimizer.update()
     loss =  sum(av_loss) / len(av_loss)
     print({'epoch': epoch, 'train_loss': loss})
     ep_loss.append({'epoch': epoch, 'loss': loss})
     # test classifier
     loss = F.sum(classifier(inputs, labels))
     print({'epoch': epoch, 'test_loss': float(loss.data)})
     test_loss.append({'epoch': epoch, 'loss': loss.data})
     av_loss = []
       
matplotlib.style.use('ggplot')
figure = plt.figure(figsize=(15,8))
ax = figure.add_subplot(111)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
e_train, l_train = [point['epoch'] for point in ep_loss], [point['loss'] for point in ep_loss]
e_test, l_test = [point['epoch'] for point in test_loss], [point['loss'] for point in test_loss]
plt.plot(e_train, l_train, 'b')
plt.plot(e_test, l_test, 'r')
plt.show() 
     