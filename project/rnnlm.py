import numpy as np

from preprocessing import getSentenceData
from rnn import Model
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
matplotlib.style.use('ggplot')

nrepoch=20
word_dim = 100
hidden_dim = 64

def run(X_train, X_test, y_train, y_test, title):
    # training phase
    rnn_b = Model(word_dim, hidden_dim)
    rnn_f = Model(word_dim, hidden_dim, mode='fa')
    rnn_df = Model(word_dim, hidden_dim, mode='dfa')
    b_loss = rnn_b.train(X_train, y_train, learning_rate=0.05, nepoch=20, evaluate_loss_after=1)
    f_loss = rnn_f.train(X_train, y_train, learning_rate=0.05, nepoch=20, evaluate_loss_after=1)
    df_loss = rnn_df.train(X_train, y_train, learning_rate=0.05, nepoch=20, evaluate_loss_after=1)
    b_loss = [loss for seen, loss in b_loss]
    f_loss = [loss for seen, loss in f_loss]
    df_loss= [loss for seen, loss in df_loss]
    
    # Plot the results
    plt.figure(figsize = (15, 8))
    plt.xticks(range(1, nrepoch + 1))
    backprop, = plt.plot(b_loss, label='backprop')
    feedback, = plt.plot(f_loss, label='feedback')
    dfeedback, = plt.plot(df_loss, label='direct_feedback')
    plt.legend(handles=[backprop, feedback,dfeedback])
    plt.title(title)
    plt.show()
    #test phase
    b_test = rnn_b.calculate_total_loss(X_test,y_test)
    f_test = rnn_f.calculate_total_loss(X_test,y_test)
    df_test = rnn_df.calculate_total_loss(X_test,y_test)
    print("backprop_loss: " + str(b_test), "feedback_loss: "+ str(f_test), "direct_feedback_loss: "+ str(df_test))

# Shakespear data
X, y = getSentenceData('data/tinyshakespear.txt', word_dim)
X_train, X_test, y_train, y_test = X[:10000], X[20000:23000], y[:10000], y[20000:23000]
shakespear_title = 'Loss on Tiny Shakespear'
np.random.seed(10)
run(X_train, X_test, y_train, y_test, shakespear_title)

# Reddit data
X, y = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)
X_train, X_test, y_train, y_test = X[:5000], X[20000:21500], y[:5000], y[20000:21500]
reddit_title = 'Loss on Reddit Comments'
np.random.seed(10)
run(X_train, X_test, y_train, y_test, reddit_title)

