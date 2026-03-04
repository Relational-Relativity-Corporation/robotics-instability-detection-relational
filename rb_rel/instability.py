"""
Instability detection via the ABCRE invariant relational kernel.

Operators (declared over bounded domain D):

    A(x)[i]      = x[i] - mean(x)                               # gradient extraction
    R(x, rho)[i] = x[i] + rho*(x[(i+1)%n] - x[(i-1)%n])       # antisymmetric circulation
    C(x)[i]      = x[i] / (1 + |x[i]|)                         # bounded coherence, output in (-1,1)

Detection pipeline:

    relational_spread(Q)   = ||A(Q)||^2 / n     # population variance by construction
    relational_momentum(W) = mean(|C(R(A(W)))|) # bounded monotonic trend signal

Applied to the fleet queue depth vector Q(t) = [Q_0, ..., Q_N-1].

No scipy dependency. All operators are pure functions over declared finite fields.

Note: relational_spread equals population variance (numpy default: ddof=0).
"""

import numpy as np


# ── Primitive operators ───────────────────────────────────────

def _operator_a(field: list) -> list:
    """A: relational gradient extraction. Zero-sum by construction."""
    mean = sum(field) / len(field)
    return [x - mean for x in field]


def _operator_r(field: list, rho: float) -> list:
    """R: antisymmetric circulation. Introduces directional bias."""
    n = len(field)
    return [
        field[i] + rho * (field[(i + 1) % n] - field[(i - 1) % n])
        for i in range(n)
    ]


def _operator_c(field: list) -> list:
    """C: bounded coherence. Output in (-1, 1) by construction."""
    return [x / (1.0 + abs(x)) for x in field]


# ── Detection functions ───────────────────────────────────────

def relational_spread(field: np.ndarray) -> float:
    """
    ||A(Q)||^2 / n

    Relational variance of the fleet queue depth vector.
    Equals population variance by construction — no statistical import required.
    """
    f = field.astype(float)
    a = f - f.mean()
    return float(np.sum(a * a) / len(a))


def rolling_relational_spread(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling relational_spread over the spread time series."""
    n  = len(series)
    rv = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        rv[i] = relational_spread(series[i - window + 1 : i + 1])
    return rv


def relational_momentum(series: np.ndarray, window: int = 30,
                        rho: float = 0.4) -> np.ndarray:
    """
    Monotonic trend signal via mean(|C(R(A(W), rho))|).

    For a rising window W:
    - A centers W preserving slope
    - R amplifies forward differences via antisymmetric circulation
    - C bounds output to (-1, 1)
    - mean(|C(...)|) grows with trend strength, direction-independent

    Replaces Kendall tau. No scipy dependency.
    """
    n   = len(series)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]
        if np.any(np.isnan(seg)):
            continue
        a      = _operator_a(list(seg))
        r      = _operator_r(a, rho)
        c      = _operator_c(r)
        out[i] = float(np.mean(np.abs(c)))
    return out