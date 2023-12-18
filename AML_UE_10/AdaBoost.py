import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class AdaBoost:

    def __init__(self, base_learner, n_estimators=10):
        self.T = n_estimators
        self.base_clf = base_learner
        self.estimators = None

        self.weights_misclassified = []
        self.sum_weights = []
        self.empirical_risk = []
        self.error = []
        self.error_term = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples)
        self.estimators = []

        for _ in range(self.T):
            w = w / np.sum(w)

            h = clone(self.base_clf)
            h.fit(X, y, sample_weight=w)

            y_hat = h.predict(X)

            # Save weights (of misclassified samples)
            self.sum_weights.append(np.sum(w))
            self.weights_misclassified.append(np.sum(w[y_hat != y]))
            self.empirical_risk.append(np.mean(y_hat != y))

            e = 1 - accuracy_score(y, y_hat, sample_weight=w)
            self.error.append(e)    # For plot
            self.error_term.append(2 * np.sqrt(e * (1 - e)))    # For plot
            a = 0.5 * np.log((1 - e) / e)

            m = 1 * (y == y_hat) - 1 * (y != y_hat)
            w *= np.exp(-a * m)

            self.estimators.append((a, h))

        return self.estimators

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for a, h in self.estimators:
            pred += a * h.predict(X)
        return np.sign(pred)

    def score(self, X, y):
        y_hat = self.predict(X)
        return accuracy_score(y, y_hat)