import numpy as np
import collections


def load_data():
    x, y = [], []
    f = open("iris.data", "r", encoding="utf-8")
    for line in f:
        line = line.strip().split(",")
        if len(line) > 1:
            x.append(list(map(float, line[:-1])))
            y.append(line[-1])
    f.close()
    return np.array(x), np.array(y)


def knn(x, k):
    N, D = x.shape
    mu = np.random.randn(k, 1, D)
    x = x.reshape(1, N, D)
    dist = np.mean((x - mu) ** 2, axis=-1)
    r = np.argmin(dist, axis=0)
    r_old = -np.ones_like(r)
    while not np.isclose(r, r_old).all():
        r_old = r.copy()
        for i in range(k):
            idx = np.where(r == i)[0]
            if len(idx) > 0:
                mu[i, 0] = np.mean(x[0, idx, :], axis=0)
        dist = np.mean((x - mu) ** 2, axis=-1)
        r = np.argmin(dist, axis=0)
    return r


def get_prob_gaussian(x, mu, sigma):
    D = len(mu)
    multiplier = (2 * np.pi) ** (-D / 2) * np.abs(np.linalg.det(sigma)) ** (-0.5)
    exp = np.exp(-0.5 * ((x - mu) * np.matmul(np.linalg.inv(sigma), (x - mu).T).T).sum(axis=1))
    return multiplier * exp


def gmm(x, k):
    N, D = x.shape
    pi = np.ones((k, )) / k
    mu = np.random.randn(k, D)
    sigma = np.zeros((k, D, D))
    for i in range(k):
        for j in range(D):
            sigma[i, j, j] = 1
    r = np.zeros((k, N))
    for _ in range(10):
        for i in range(k):
            r[i] = pi[i] * get_prob_gaussian(x, mu[i], sigma[i])
        r /= np.sum(r, axis=0, keepdims=True)
        mu = np.dot(r, x) / np.sum(r, axis=1).reshape(-1, 1)
        delta = x.reshape(N, 1, D) - mu.reshape(1, k, D)  # [N, k, D]
        for i in range(k):
            r_k = r[i].reshape(N, 1, 1)
            delta_k = delta[:, i, :]  # [N, D]
            sigma[i] = (r_k * delta_k.reshape(N, D, 1) * delta_k.reshape(N, 1, D)).mean(axis=0)
        for p in range(k):
            for q in range(D):
                sigma[p, q, q] += 1e-4
        pi = r.sum(axis=1) / N
    return np.argmax(r, axis=0)


def main():
    x, y = load_data()
    for k in [3, 4, 5]:
        z = knn(x, k)
        purity = get_purity(y, z)
        print(f"KNN: k={k}, purity={purity}")
    for k in [3, 4, 5]:
        z = gmm(x, k)
        purity = get_purity(y, z)
        print(f"GMM: k={k}, purity={purity}")


def get_purity(y, pred):
    labels = list(set(pred))
    purity = 0
    for label in labels:
        idx = np.where(pred == label)[0]
        y_idx = y[idx]
        counter = collections.Counter(y_idx)
        max_count = -1
        for (k, v) in counter.items():
            if v > max_count:
                max_count = v
        purity += 1 / len(labels) * max_count / len(idx)
    return purity


if __name__ == "__main__":
    main()
