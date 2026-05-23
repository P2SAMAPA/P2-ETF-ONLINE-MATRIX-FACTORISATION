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
        d = len(x)
        if self.mean is None:
            self.mean = x.copy()
            self.cov = np.zeros((d, d))
            self.n_samples = 1
            self.eigenvectors = np.random.randn(d, self.n_components)
            self.eigenvectors /= np.linalg.norm(self.eigenvectors, axis=0)
            self.eigenvalues = np.zeros(self.n_components)
            return

        old_mean = self.mean.copy()
        self.mean = self.forget * self.mean + (1 - self.forget) * x
        delta = x - old_mean
        self.cov = self.forget * self.cov + (1 - self.forget) * np.outer(delta, delta)
        self.n_samples += 1

        # Oja's rule update (one power iteration per sample)
        for i in range(self.n_components):
            v = self.eigenvectors[:, i]
            v_new = self.cov @ v
            for j in range(i):
                v_new -= (v_new @ self.eigenvectors[:, j]) * self.eigenvectors[:, j]
            norm = np.linalg.norm(v_new)
            if norm > 0:
                v_new /= norm
            self.eigenvectors[:, i] = v_new
            self.eigenvalues[i] = v_new @ (self.cov @ v_new)

    def get_loadings(self):
        return self.eigenvectors[:, 0]  # first component loadings

def online_pca_score(returns_df, window, n_components=3, learning_rate=0.01, forget_factor=0.95):
    """
    Process the last `window` days of returns incrementally.
    Returns a dict {ticker: absolute loading in first PC}.
    """
    if len(returns_df) < window:
        return None
    data = returns_df.iloc[-window:].values  # (T, d)
    pca = IncrementalPCA(n_components, learning_rate, forget_factor)
    for i in range(data.shape[0]):
        pca.update(data[i])
    loadings = pca.get_loadings()
    tickers = returns_df.columns.tolist()
    scores = {tickers[i]: abs(loadings[i]) for i in range(len(tickers))}
    return scores
