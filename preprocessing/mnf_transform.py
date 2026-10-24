import numpy as np
import rasterio


def compute_mnf(data_hwb: np.ndarray, nodata: float, num_components: int) -> np.ndarray:
    """
    Minimum Noise Fraction transform on a hyperspectral image.

    Noise is estimated from horizontal spatial differences between adjacent valid
    pixels.  The first MNF component has the highest signal-to-noise ratio.

    Parameters
    ----------
    data_hwb      : (H, W, B) float32 array, nodata pixels marked with `nodata`
    nodata        : scalar sentinel value for invalid pixels
    num_components: number of MNF components to return

    Returns
    -------
    (H, W, num_components) float32; nodata pixels keep the nodata value.
    """
    H, W, B = data_hwb.shape

    valid_2d = ~np.all(data_hwb == nodata, axis=2)          # (H, W) bool

    # ── Stage 1: noise covariance via horizontal spatial differences ──────────
    # Both neighbours must be valid for the difference to be meaningful.
    diff = data_hwb[:, 1:, :] - data_hwb[:, :-1, :]        # (H, W-1, B)
    both_valid = valid_2d[:, 1:] & valid_2d[:, :-1]        # (H, W-1)
    valid_diff = diff[both_valid].astype(np.float64)        # (N_diffs, B)

    if len(valid_diff) < B + 1:
        raise ValueError(
            f"Not enough valid pixel pairs for MNF noise estimation "
            f"(got {len(valid_diff)}, need at least {B + 1})."
        )

    # Noise covariance: divide by 2 because diff of two equal-variance signals
    # has twice the variance of the original noise.
    noise_cov = np.cov(valid_diff.T) / 2.0                 # (B, B)

    evals_n, evecs_n = np.linalg.eigh(noise_cov)
    evals_n = np.clip(evals_n, 1e-10, None)
    # Whitening transform A maps original space to noise-whitened space.
    A = evecs_n @ np.diag(evals_n ** -0.5)                 # (B, B)

    # ── Stage 2: PCA on noise-whitened signal ────────────────────────────────
    flat = data_hwb.reshape(-1, B).astype(np.float64)      # (H*W, B)
    valid_mask = valid_2d.ravel()                           # (H*W,) bool
    valid_data = flat[valid_mask]                           # (N_valid, B)

    # Subsample for covariance estimation to cap memory use.
    MAX_SAMPLES = 100_000
    if len(valid_data) > MAX_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(valid_data), MAX_SAMPLES, replace=False)
        sample = valid_data[idx]
    else:
        sample = valid_data

    whitened_sample = sample @ A                            # (N_sample, B)
    sig_cov = np.cov(whitened_sample.T)                    # (B, B)

    evals_s, evecs_s = np.linalg.eigh(sig_cov)
    order = np.argsort(evals_s)[::-1]                      # descending SNR
    evecs_s = evecs_s[:, order]

    # Combined forward transform: original → MNF space (B → num_components)
    mnf_T = A @ evecs_s[:, :num_components]                # (B, num_components)

    # Apply to all valid pixels
    mnf_valid = (valid_data @ mnf_T).astype(np.float32)   # (N_valid, K)

    # Place back into full spatial grid
    mnf_full = np.full((H * W, num_components), nodata, dtype=np.float32)
    mnf_full[valid_mask] = mnf_valid

    return mnf_full.reshape(H, W, num_components)


def append_mnf_to_tif(tifpath: str, num_components: int = 10) -> None:
    """
    Compute MNF on the stacked TIF at `tifpath` and overwrite it with
    the `num_components` most informative MNF bands appended at the end.
    """
    with rasterio.open(tifpath) as src:
        data    = src.read().astype(np.float32)  # (B, H, W)
        profile = src.profile.copy()
        nodata  = src.nodata if src.nodata is not None else -9999.0

    B, H, W = data.shape
    data_hwb = np.transpose(data, (1, 2, 0))               # (H, W, B)

    mnf_hwb  = compute_mnf(data_hwb, nodata, num_components)   # (H, W, K)
    mnf_bands = np.transpose(mnf_hwb, (2, 0, 1))           # (K, H, W)

    stacked = np.concatenate([data, mnf_bands], axis=0)    # (B+K, H, W)
    profile.update(count=B + num_components)

    with rasterio.open(tifpath, 'w', **profile) as dst:
        dst.write(stacked)
