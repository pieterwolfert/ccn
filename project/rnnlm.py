import numpy as np

from preprocessing import getSentenceData
from rnn import Model
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
matplotlib.style.use('ggplot')

nrepoch=20
word_dim = 8000
hidden_dim = 100
X, y = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)
X_train, X_test, y_train, y_test = X[:20000], X[20000:-1], y[:20000], y[20000:-1]
np.random.seed(10)

# training phase
rnn_b = Model(word_dim, hidden_dim)
rnn_f = Model(word_dim, hidden_dim, feedback=True)
#rnn_df = Model(word_dim, hidden_dim, dfeedback=True)
b_loss = rnn_b.train(X_train[18000:18100], y_train[18000:18100], learning_rate=0.05, nepoch=20, evaluate_loss_after=1)
f_loss = rnn_f.train(X_train[18000:18100], y_train[18000:18100], learning_rate=0.05, nepoch=20, evaluate_loss_after=1)
b_loss = [loss for seen, loss in b_loss]
f_loss = [loss for seen, loss in f_loss]
backprop, = plt.plot(b_loss, label='backprop')
feedback, = plt.plot(f_loss, label='feedback')
#dfeedback, = plt.plot(f_loss, label='direct_feedback')
plt.legend(handles=[backprop, feedback])

# test phase
b_test = rnn_b.calculate_total_loss(X_test,y_test)
f_test = rnn_f.calculate_total_loss(X_test,y_test)
#df_test = rnn_df.calculate_total_loss(X_test,y_test)
print("backprop_loss: " + str(b_test), "feedback_loss: "+ str(f_test))