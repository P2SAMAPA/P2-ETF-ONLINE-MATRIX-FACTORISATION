# Online Matrix Factorisation Engine

Implements incremental PCA using Oja's rule and power iteration. Updates factor loadings online in O(d²) per new day, making it suitable for non‑stationary regimes. The score is the absolute loading of each ETF in the first principal component. Multi‑window evaluation selects the best window per ETF.

- **Algorithm:** Incremental PCA with forgetting factor
- **Update rule:** Oja's rule + covariance tracking
- **Score:** |loading in first component| (higher = more influential)
- **Windows:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
