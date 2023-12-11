import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CarlitoBoost:
    def __init__(self, T=50, base_learner=DecisionTreeClassifier):
        self.T = T
        self.base_learner=base_learner
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        data_weights = np.ones(n_samples) * (1/n_samples)   # Initialize weights as 1/n
        # Implement Resampling!

        for t in range(self.T):
            # Reweight data
            X *= data_weights

            # Choose t-th base learner
            base_learner_t = self.base_learner
            base_learner_t.fit(X)
            predictions = base_learner_t.predict(X)

            # Compute empirical risk $\epsilon_{t}^{*}$
            misclassifications = predictions != y
            empirical_risk = np.dot(data_weights, misclassifications)

            alpha_t = 0.5 * np.log((1-empirical_risk) / empirical_risk)

            # Add t-th itertation to the ensemble
            self.estimators.append(base_learner_t)
            self.estimator_weights.append(alpha_t)

            # Normalize weights to sum to one
            data_weights = (data_weights * np.exp(-alpha_t * y * predictions)) / np.sum(data_weights)

    def predict(self, X):
        estimator_predictions = np.array(clf.predict(X) for clf in self.estimators)
        pred = np.dot(self.estimator_weights, estimator_predictions)
        return pred
