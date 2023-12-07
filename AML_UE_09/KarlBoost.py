from sklearn.tree import DecisionTreeClassifier
import numpy as np

class KarlBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, sample_weight=weights)
            predictions = clf.predict(X)

            misclassified = predictions != y
            error = np.dot(weights, misclassified) / weights.sum()
            alpha = np.log((1 - error) / error) / 2

            weights *= np.exp(alpha * misclassified * ((weights > 0) | (alpha < 0)))
            weights /= weights.sum()

            self.estimators.append(clf)
            self.estimator_weights.append(alpha)

    def predict(self, X):
        clf_predictions = np.array([clf.predict(X) for clf in self.estimators])
        final_prediction = np.sign(np.dot(self.estimator_weights, clf_predictions))
        return final_prediction
