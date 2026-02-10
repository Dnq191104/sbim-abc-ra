import numpy as np


def mmd_rbf_1d(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    x = x.reshape(-1, 1).astype(float)
    y = y.reshape(-1, 1).astype(float)

    if sigma is None:
        z = np.vstack([x, y])
        if z.shape[0] > 2000:
            idx = np.random.choice(z.shape[0], 2000, replace=False)
            z = z[idx]
        d2 = (z - z.T) ** 2
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med / 2) if np.isfinite(med) and med > 0 else 1.0

    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)

    def k(a, b):
        return np.exp(-gamma * (a - b.T) ** 2)

    Kxx = k(x, x)
    Kyy = k(y, y)
    Kxy = k(x, y)

    n = x.shape[0]
    m = y.shape[0]
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (
        Kxx.sum() / (n * (n - 1) + 1e-12)
        + Kyy.sum() / (m * (m - 1) + 1e-12)
        - 2.0 * Kxy.mean()
    )
    return float(max(mmd2, 0.0))


def mean_mmd_over_dims(theta: np.ndarray, theta_ref: np.ndarray) -> dict:
    d = theta.shape[1]
    per = []
    for j in range(d):
        per.append(mmd_rbf_1d(theta[:, j], theta_ref[:, j]))
    per = np.array(per, dtype=float)
    return {
        "mmd_mean": float(np.mean(per)),
        "mmd_median": float(np.median(per)),
        "mmd_per_dim": per.tolist(),
    }


