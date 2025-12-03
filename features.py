import numpy as np

def post_decision_state(s, A, x):
    return s + A.T.dot(x)

def phi_polynomial(sx, degree=2, s_max=70):
    sx = np.array(sx, dtype=float)
    feats = [0.1]
    for d in range(1, degree+1):
        feats.extend((sx**d).tolist())
    return np.array(feats, dtype=float)

def phi_fourier(sx_norm, freq_vectors=None, per_dim_freqs=0):
    """
    Flexible Fourier-style feature generator.

    Modes:
    - If freq_vectors is provided (list/array of vectors), computes:
        phi = [0.1] + cos(pi * freqs.dot(sx_norm)) for each freq vector
      (backwards-compatible with previous usage).
    - Else if per_dim_freqs > 0, for each dimension j and frequency k in 1..per_dim_freqs
      produce cos(pi*k*sx_norm[j]) and sin(pi*k*sx_norm[j]) features (per-dim multi-frequency).
    - Else fallback: include bias + normalized sx itself as features.

    Returns:
      1D numpy array of features.
    """
    sx = np.array(sx_norm, dtype=float)
    feats = [0.1]
    if freq_vectors is not None:
        freqs = np.array(freq_vectors, dtype=float)
        # freqs.dot(sx) will produce one value per freq vector
        vals = np.cos(np.pi * freqs.dot(sx))
        feats.extend(vals.tolist())
    elif per_dim_freqs and per_dim_freqs > 0:
        for k in range(1, per_dim_freqs + 1):
            feats.extend(np.cos(np.pi * k * sx).tolist())
            feats.extend(np.sin(np.pi * k * sx).tolist())
    else:
        # fallback: include normalized per-dimension inventory as features
        feats.extend(sx.tolist())
    return np.array(feats, dtype=float)

def normalize_sx(sx, s_max):
    return np.array(sx, dtype=float) / float(s_max)