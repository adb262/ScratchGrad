import math
import numpy as np


class BatchNorm:
    def __init__(self, gamma: float, beta: float):
        self.gamma = gamma
        self.beta = beta
        self.cache = None

    def forward(self, batch_X):
        # normalize the columns
        totals = [0] * len(batch_X[0])
        for i in range(len(batch_X)):
            for j in range(len(batch_X[i])):
                totals[j] += batch_X[i][j]

        means = [x / len(batch_X) for x in totals]
        stdev_cumulator = [0] * len(means)
        for i in range(len(batch_X)):
            for j in range(len(batch_X[i])):
                stdev_cumulator[j] += (batch_X[i][j] - means[j]) ** 2

        stds = [math.sqrt(x) / len(batch_X) for x in stdev_cumulator]
        centered_x = [[(x[j] - means[j]) for j in range(len(x))] for x in batch_X]
        normalized_x = [[x[j] / (stds[j]**2 + 1e-5) for j in range(len(x))] for x in centered_x]

        scaled_x = [[(x[j] * self.gamma[j]) + self.beta[j] for j in range(len(x))] for x in normalized_x]
        self.cache = (normalized_x, centered_x, means, stds)
        return scaled_x

    def backward(self, dout: list[list[float]]):
        N, D = len(dout), len(dout[0])
        normalized_x, centered_x, means, stds = self.cache

        # intermediate partial derivatives
        dxhat = dout * self.gamma
        dxhat_sum = sum(dxhat)

        dbeta = sum(dout)

        # final partial derivatives
        dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0)
                                   - x_hat*np.sum(dxhat*x_hat, axis=0))
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(x_hat*dout, axis=0)
