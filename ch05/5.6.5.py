'''
部分输出：
GD      acc:319/351 = 0.9088319088319088, iter = 1647
NW      acc:319/351 = 0.9088319088319088, iter = 8
SGD     acc:317/351 = 0.9031339031339032, iter = 3000 未收敛

牛顿法最快达到收敛，最速梯度下降法次之，随机梯度下降法在以对数似然函数变化量绝对值小于1e-6作为收敛判定标准时收敛最慢，故在此限制其仅3000轮次。
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

def grad_desc(X, y, lr=0.01, epsilon=1e-6):

    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    while (np.abs(ll_cur - ll_former) > epsilon):
        ll_former = ll_cur

        grad = get_grad(w, X, y)

        w = w - lr * grad

        ll_cur = log_likelihood(w, X, y)
        print("it:{}, ll:{}".format(it, ll_cur))
        ll_trace.append(ll_cur)
        it += 1
    return w, ll_trace

def newton_method(X, y, epsilon=1e-6):
    
    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    while (np.abs(ll_cur - ll_former) > epsilon):
        ll_former = ll_cur
        
        h = _sigmoid(np.dot(X, w))

        grad = np.dot(X.T, y - h)
        hessian = np.dot(X.T * h * (1 - h), X)

        w  += np.dot(np.linalg.inv(hessian + 0.001 * np.identity(hessian.shape[0])), grad)

        ll_cur = log_likelihood(w, X, y)
        print("it:{}, ll:{}".format(it, ll_cur))
        ll_trace.append(ll_cur)
        it += 1
    return w, ll_trace


def stocha_grad_desc(X, y, lr=0.01, epsilon=1e-6, batch_size=50, max_iter=3000):
    w = np.zeros(X.shape[1])
    ll_cur = log_likelihood(w, X, y)
    ll_former = 1e6
    it = 0
    ll_trace = list()
    while (np.abs(ll_cur - ll_former) > epsilon) and (it < max_iter):
        ll_former = ll_cur

        for _ in range(X.shape[0] // batch_size):
            sample_idx = np.random.choice(X.shape[0], size=batch_size, replace=False)
            grad = get_grad(w, X[sample_idx, :], y[sample_idx])

            w = w - lr * grad

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

    w_gd, ll_trace_gd = grad_desc(X, y, lr=0.01)

    w_nw, ll_trace_nw = newton_method(X, y)

    w_sgd, ll_trace_sgd = stocha_grad_desc(X, y, batch_size=100, max_iter=3000)

    print("GD\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_gd)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_gd)) > 0.5) == y) / y.shape[0], len(ll_trace_gd)))
    
    print("NW\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_nw)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_nw)) > 0.5) == y) / y.shape[0], len(ll_trace_nw)))
    
    print("SGD\tacc:{}/{} = {}, iter = {}".format(np.sum((_sigmoid(np.dot(X, w_sgd)) > 0.5) == y), y.shape[0],
        np.sum((_sigmoid(np.dot(X, w_sgd)) > 0.5) == y) / y.shape[0], len(ll_trace_sgd)))


    plt.plot(ll_trace_gd, label='gd')
    plt.plot(ll_trace_nw, label='nw')
    plt.plot(ll_trace_sgd, label='sgd')
    plt.legend()
    plt.show()
    plt.close()

