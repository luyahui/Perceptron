import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """
    eta : learning probability
    n_iter : iteration times
    w_ : weight vector
    errors : errors for each iteration
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            error = 0
            for x, target in zip(X, y):
                predicted = 1 if self.net_input(x) >= 0.0 else -1
                update = self.eta * (target - predicted)
                self.w_[1:] += update * x
                self.w_[0] += update
                error += 1 if (update != 0) else 0
            self.errors.append(error)
        pass

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


if __name__ == "__main__":
    # read data from a csv file
    file = "iris.csv"
    df = pd.read_csv(file)

    X = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == "setosa", -1, 1)

    # training
    ppn = Perceptron()
    ppn.fit(X, y)

    def plot_decision_regions(X, y, classifier, resolution=0.1):
        markers = ("x", "o", "v")
        colors = ("blue", "red", "cyan")
        cmap = ListedColormap(colors[: len(np.unique(y))])

        x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
        x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
        )

        z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(
                X[y == cl, 0],
                X[y == cl, 1],
                alpha=0.8,
                c=cmap(idx),
                marker=markers[idx],
                label=cl,
            )

    plot_decision_regions(X, y, ppn)
    plt.xlabel('Sepal length')
    plt.ylabel('Petal length')
    plt.legend()
    plt.show()
