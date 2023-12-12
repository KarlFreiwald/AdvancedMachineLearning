import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CarlitoBoost:
    def __init__(self, T=50, base_learner=DecisionTreeClassifier, base_learner_args=None):
        self.T = T
        self.base_learner = base_learner
        self.base_learner_args = base_learner_args if base_learner_args is not None else {}
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y, sample_weights: np.ndarray = None):
        n_samples = X.shape[0]

        if sample_weights is not None:
            if sample_weights.shape[0] != X.shape[0]:
                raise ValueError(f"{sample_weights.shape[0]=} does not match {X.shape[0]=}.")
            data_probabilities = sample_weights / np.sum(sample_weights)    # Normalize sample_weights
        else:
            # Initialize weights as 1/n
            data_probabilities = np.full(shape=n_samples, fill_value=1 / n_samples)

        indices = np.arange(n_samples)  # Create an array of indices

        for t in range(self.T):
            # Sample indices based on the updated probabilities to do Resampling
            sampled_indices = np.random.choice(indices, size=n_samples, replace=True, p=data_probabilities)

            # Choose t-th base learner
            base_learner_t = self.base_learner(**self.base_learner_args)
            base_learner_t.fit(X[sampled_indices], y[sampled_indices]) # Fit on resampled data set.
            predictions = base_learner_t.predict(X) # Predict on normal data set!

            # Compute empirical risk
            misclassifications = predictions != y
            empirical_risk = np.dot(data_probabilities, misclassifications)
            small_constant = 0.001 # Adding a small constant to prevent alphas to be infinite
            alpha_t = np.log((1 - empirical_risk) / (empirical_risk + small_constant)) / 2

            # Add t-th itertation to the ensemble
            self.estimators.append(base_learner_t)
            self.estimator_weights.append(alpha_t)

            # Normalize weights to sum to one
            data_probabilities = (data_probabilities * np.exp(-alpha_t * y * predictions))
            data_probabilities /= np.sum(data_probabilities)  # Normalize weights

    def predict(self, X):
        estimator_predictions = np.array([clf.predict(X) for clf in self.estimators])
        final_prediction = np.sign(np.dot(self.estimator_weights, estimator_predictions))
        return final_prediction
