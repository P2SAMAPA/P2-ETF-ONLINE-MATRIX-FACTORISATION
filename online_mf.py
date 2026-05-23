import numpy as np

class IncrementalPCA:
    def __init__(self, n_components=3, learning_rate=0.01, forget_factor=0.95):
        self.n_components = n_components
        self.lr = learning_rate
        self.forget = forget_factor
        self.mean = None
        self.cov = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.n_samples = 0

    def update(self, x):
        """
        x: new observation (1D numpy array of length d)
        """
        d = len(x)
        if self.mean is None:
            self.mean = x.copy()
            self.cov = np.zeros((d, d))
            self.n_samples = 1
            # Initialise eigenvectors randomly
            self.eigenvectors = np.random.randn(d, self.n_components)
            self.eigenvectors /= np.linalg.norm(self.eigenvectors, axis=0)
            self.eigenvalues = np.zeros(self.n_components)
            return

        # Update mean
        old_mean = self.mean.copy()
        self.mean = self.forget * self.mean + (1 - self.forget) * x
        # Update covariance with forgetting
        delta = x - old_mean
        self.cov = self.forget * self.cov + (1 - self.forget) * np.outer(delta, delta)
        self.n_samples += 1

        # Update eigenvectors using Oja's rule (power iteration)
        for i in range(self.n_components):
            v = self.eigenvectors[:, i]
            # One step of power iteration: v' = C v
            v_new = self.cov @ v
            # Orthogonalise against previous components
            for j in range(i):
                v_new -= (v_new @ self.eigenvectors[:, j]) * self.eigenvectors[:, j]
            # Normalise
            norm = np.linalg.norm(v_new)
            if norm > 0:
                v_new /= norm
            self.eigenvectors[:, i] = v_new
            # Update eigenvalue (Rayleigh quotient)
            self.eigenvalues[i] = v_new @ (self.cov @ v_new)

    def get_weights(self):
        """
        Return the first principal component (loadings for ETFs) as a vector.
        Higher absolute loading means more important.
        """
        if self.eigenvectors is None:
            return np.zeros(self.cov.shape[0])
        return self.eigenvectors[:, 0]  # first component

def online_pca_score(returns_series, n_components=3, learning_rate=0.01, forget_factor=0.95):
    """
    Run incremental PCA on a return series (pandas Series) over time.
    Return the weight of the ETF in the first principal component (absolute value) as the score.
    """
    # Convert to numpy array (daily returns)
    data = returns_series.values.reshape(-1, 1)  # we need a 2D array
    # But incremental PCA expects each observation as a vector of features.
    # Here we treat each ETF separately, so we have a univariate time series.
    # That's not appropriate. Instead, we need to consider all ETFs together.
    # This function should work on the entire universe, not per ETF.
    # So we need to pass the entire returns matrix for the window and update incrementally.
    # Let's redesign: For a given window, we initialise IncrementalPCA with the first `init` days,
    # then update with the rest, and finally compute the weight for each ETF from the first eigenvector.
    pass
