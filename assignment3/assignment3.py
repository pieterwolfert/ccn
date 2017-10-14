# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import chainer
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
import chainer.training
import matplotlib.pyplot as plt
import matplotlib
from chainer import iterators


       
class BPTTUpdater(chainer.training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, rnn):
        super(BPTTUpdater, self).__init__(train_iter, optimizer)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        train_loss = []
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        count = 0
        loss = 0
        # Progress the dataset iterator for bprop_len 
        for i in range(self.bprop_len):
            count += 1
            # Get the next batch 
            batch = train_iter.__next__()
            
            x, t = self.converter(batch)
            rnn.reset_state()
            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
        
            if count % 10 == 0:
                optimizer.target.cleargrads()  # Clear the parameter gradients
                loss.backward()  # Backprop
                loss.unchain_backward()
                optimizer.update()  # Update the parameters
                train_loss.append(loss)
                        
        return train_loss
        
        
class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__()
        with self.init_scope():
            self.lstm = L.LSTM(None,1) 
            self.linear = L.Linear(None,1)
                
    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x):
        # Given the current input compute the sum
        h = self.lstm(x)
        y = self.linear(h)
        return y
                

class Regressor(Chain):
    """

    Regressor chain on top of Neural Network chain 
    for accuracy evaluation and prediction
 
    """
    def __init__(self, predictor):
        super(Regressor, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        return loss


def compute_loss(inputs, labels, model):
    loss = model(inputs, labels)
    return loss


def create_data(n=3000):

    X = np.random.rand(n,1).astype('float32')
    T = np.sum(np.hstack((X[0:-1],X[1:])),axis=1)
    T = np.hstack([0, T[0:]]).astype('float32')
    T = T.reshape([n,1])

    return TupleDataset(X, T)


def visualize_loss(train_loss, test_loss, nr_epochs=20):
    matplotlib.style.use('ggplot')
    figure = plt.figure(figsize=(15, 8))
    ax = figure.add_subplot(111)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    xticks = np.arange(1, nr_epochs+1, 5)
    ax.set_xticks(xticks)
    epochs = range(1, nr_epochs + 1)
    train, = plt.plot(epochs, train_loss, 'b', label='train loss')
    test, = plt.plot(epochs, test_loss, 'r', label='test loss')
    plt.legend([train, test], ['train loss', 'test loss'])
    plt.show()

train_data = create_data()
test_data = create_data()
inputs, labels = np.array([tup[0] for tup in test_data]), np.array([tup[1] for tup in test_data])

train_iter = iterators.SerialIterator(train_data, 1)
rnn = RNN()
regressor = Regressor(rnn)
optimizer = chainer.optimizers.SGD(lr = 0.05)
optimizer.setup(regressor)
updater = BPTTUpdater(train_iter, optimizer,3000, rnn)
train_loss, test_loss = [], []


train_loss = updater.update_core()
rnn.reset_state()
test_loss.append(compute_loss(inputs, labels, regressor).data)
    
rnn.reset_state()
#visualize_loss(train_loss, test_loss, nr_epochs=3000)
new_inputs = inputs[1:100]
pred_inputs = rnn(new_inputs).data
actual_inputs = labels[1:100]
pred = plt.plot(pred_inputs, 'b', label = "predicted inputs")
actual = plt.plot(actual_inputs, 'r', label = "actual inputs")
plt.legend([pred, actual],['predicted inputs', 'actual inputs'])

plt.show()
print(labels.shape)
print(inputs.shape)
