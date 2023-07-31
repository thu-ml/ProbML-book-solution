'''
参考：
Adagrad：https://dmitrijskass.netlify.app/2021/04/15/adagrad-adaptive-gradient-algorithm/
RMSProp & Adam：https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be
'''

import numpy as np
import csv
import matplotlib.pyplot as plt


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(w, X, y):
    h = _sigmoid(np.dot(X, w))
    ll = np.sum(y * np.log(h + 1e-6) + (1-y) * np.log(1-h + 1e-6))
    return ll

def get_grad(w, X, y):
    err = _sigmoid(np.dot(X, w)) - y
    grad = np.dot(X.T, err)
    return grad

def sgd_momentum(X, y, lr=0.01, gamma=0.9, epsilon=1e-3, batch_size=50, max_iter=3000):
    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    v_w = 0
    while (np.abs(ll_cur - ll_former) > epsilon) and (it < max_iter):
        ll_former = ll_cur

        for _ in range(X.shape[0] // batch_size):
            sample_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            grad = get_grad(w, X[sample_idx, :], y[sample_idx])

            v_w = gamma * v_w + lr * grad
            w = w - v_w

        ll_cur = log_likelihood(w, X, y)
        print("it:{}, ll:{}".format(it, ll_cur))
        ll_trace.append(ll_cur)
        it += 1
    return w, ll_trace

def adagrad(X, y, lr=0.01, epsilon=1e-3, batch_size=50, max_iter=3000):
    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    G = np.zeros(X.shape[1])
    while (np.abs(ll_cur - ll_former) > epsilon) and (it < max_iter):
        ll_former = ll_cur

        for _ in range(X.shape[0] // batch_size):
            sample_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            grad = get_grad(w, X[sample_idx, :], y[sample_idx])

            G += grad ** 2
            step = lr / (np.sqrt(G + 1e-6)) * grad
            w -= step

        ll_cur = log_likelihood(w, X, y)
        print("it:{}, ll:{}".format(it, ll_cur))
        ll_trace.append(ll_cur)
        it += 1
    return w, ll_trace

def rmsprop(X, y, lr=0.01, gamma=0.9, epsilon=1e-3, batch_size=50, max_iter=3000):
    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    v_w = 0
    while (np.abs(ll_cur - ll_former) > epsilon) and (it < max_iter):
        ll_former = ll_cur

        for _ in range(X.shape[0] // batch_size):
            sample_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            grad = get_grad(w, X[sample_idx, :], y[sample_idx])

            v_w = gamma * v_w + (1 - gamma) * (grad ** 2)
            w = w - lr / (np.sqrt(v_w + 1e-6)) * grad

        ll_cur = log_likelihood(w, X, y)
        print("it:{}, ll:{}".format(it, ll_cur))
        ll_trace.append(ll_cur)
        it += 1
    return w, ll_trace

def adam(X, y, lr=0.01, gamma1=0.9, gamma2 = 0.9, epsilon=1e-3, batch_size=50, max_iter=3000):
    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    m_w = 0.
    v_w = 0.
    gamma1_decay = 1.
    gamma2_decay = 1.
    while (np.abs(ll_cur - ll_former) > epsilon) and (it < max_iter):
        ll_former = ll_cur

        gamma1_decay *= gamma1
        gamma2_decay *= gamma2

        for _ in range(X.shape[0] // batch_size):
            sample_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            grad = get_grad(w, X[sample_idx, :], y[sample_idx])

            m_w = gamma1 * m_w + (1 - gamma1) * grad
            v_w = gamma2 * v_w + (1 - gamma2) * (grad ** 2)
            
            m_hat = m_w / (1 - gamma1_decay)
            v_hat = v_w / (1 - gamma2_decay)

            w = w - lr * m_hat / (np.sqrt(v_hat + 1e-6))

        ll_cur = log_likelihood(w, X, y)
        print("it:{}, ll:{}".format(it, ll_cur))
        ll_trace.append(ll_cur)
        it += 1
    return w, ll_trace

if __name__ == '__main__':

    X = np.zeros((351, 32), dtype='float')
    y = np.zeros((351,), dtype='bool')

    with open("ionosphere.data", 'r') as data:
        reader = csv.reader(data)
        for i, row in enumerate(reader):
            X[i] = [float(item) for item in row[2:-1]]
            y[i] = (row[-1] == 'g')

    X = np.concatenate((X, np.ones((351, 1))), axis=1)
    print(X.shape)

    w_sgdm, ll_trace_sgdm = sgd_momentum(X, y, lr=0.01, gamma=0.5, epsilon=1e-3, batch_size=50)

    w_adagrad, ll_trace_adagrad = adagrad(X, y, lr=0.01, epsilon=1e-3, batch_size=50, max_iter=5000)

    w_rmsprop, ll_trace_rmsprop = rmsprop(X, y, lr=0.01, gamma=0.5, batch_size=50, max_iter=3000)

    w_adam, ll_trace_adam = adam(X, y, lr=0.01, gamma1=0.8, gamma2=0.8, batch_size=50, max_iter=3000)

    print("SGD momentum\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_sgdm)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_sgdm)) > 0.5) == y) / y.shape[0], len(ll_trace_sgdm)))
    
    print("Adagrad\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_adagrad)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_adagrad)) > 0.5) == y) / y.shape[0], len(ll_trace_adagrad)))
    
    print("RMSProp\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_rmsprop)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_rmsprop)) > 0.5) == y) / y.shape[0], len(ll_trace_rmsprop)))

    print("Adam\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_adam)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_adam)) > 0.5) == y) / y.shape[0], len(ll_trace_adam)))


    plt.plot(ll_trace_sgdm, label='sgdm')
    plt.plot(ll_trace_adagrad, label='ada')
    plt.plot(ll_trace_rmsprop, label='rmsprop')
    plt.plot(ll_trace_adam, label='adam')
    plt.legend()
    plt.show()
    plt.close()
