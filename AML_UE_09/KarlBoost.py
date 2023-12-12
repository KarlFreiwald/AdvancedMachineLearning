from sklearn.tree import DecisionTreeClassifier
import numpy as np

class KarlBoost:
    def __init__(self, T=50):
        self.T = T
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1 / n_samples)

        for t in range(self.T):
            base_learner_t = DecisionTreeClassifier(max_depth=1)
            base_learner_t.fit(X, y, sample_weight=weights)
            predictions = base_learner_t.predict(X)

            misclassified = predictions != y
            # "/ weights.sum()" actually not necessary as =1
            error = np.dot(weights, misclassified) / weights.sum()
            alpha_t = np.log((1 - error) / error) / 2

            weights *= np.exp(alpha_t * misclassified * ((weights > 0) | (alpha_t < 0)))
            weights /= weights.sum()

            self.estimators.append(base_learner_t)
            self.estimator_weights.append(alpha_t)

    def predict(self, X):
        clf_predictions = np.array([clf.predict(X) for clf in self.estimators])
        final_prediction = np.sign(np.dot(self.estimator_weights, clf_predictions))
        return final_prediction
