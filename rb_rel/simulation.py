"""
Task-queue fleet simulation.

Governing equations:

    capacity_i(t)  = PROCESSING_RATE * e_i(t)
    arrivals_i(t)  = (capacity_i + EPS) / sum(capacity_j + EPS) * GLOBAL_TASK_RATE
    processed_i(t) = min(Q_i(t), capacity_i(t))
    Q_i(t+1)       = Q_i(t) + arrivals_i(t) - processed_i(t) + diffusion

Baseline property:
    At e_i = 1 for all i:
        sum(arrivals_i) = GLOBAL_TASK_RATE
        sum(processed_i) = N * PROCESSING_RATE = GLOBAL_TASK_RATE
        => delta Q = 0 for all i
"""

import numpy as np
from rb_rel.network import GLOBAL_TASK_RATE, PROCESSING_RATE, EPS

ALPHA = 0.10


def _dispatch(efficiency: np.ndarray) -> np.ndarray:
    """Centralized dispatcher with epsilon stabilization."""
    capacity = PROCESSING_RATE * efficiency
    weights  = capacity + EPS
    return (weights / weights.sum()) * GLOBAL_TASK_RATE


def _process(queue: np.ndarray, efficiency: np.ndarray) -> np.ndarray:
    """Each robot processes up to its current capacity, bounded by queue depth."""
    return np.minimum(queue, PROCESSING_RATE * efficiency)


def _diffuse(queue: np.ndarray, neighbors: list) -> np.ndarray:
    """Local ring diffusion — Laplacian form."""
    new_q = queue.copy()
    for i, nbrs in enumerate(neighbors):
        delta = sum(queue[j] - queue[i] for j in nbrs)
        new_q[i] += ALPHA * delta
    return np.maximum(0.0, new_q)


def simulate(params: dict, neighbors: list, T: int = 300,
             disruption_fn=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Run fleet simulation for T timesteps.

    Returns
    -------
    q_history : (T, N) array — queue depth per robot per timestep
    e_history : (T, N) array — efficiency per robot per timestep
    """
    queue      = params['queue'].copy()
    efficiency = params['efficiency'].copy()
    N          = len(queue)
    q_history  = np.zeros((T, N))
    e_history  = np.zeros((T, N))

    for t in range(T):
        if disruption_fn is not None:
            efficiency = disruption_fn(efficiency, t)
        arrivals        = _dispatch(efficiency)
        processed       = _process(queue, efficiency)
        queue           = queue + arrivals - processed
        queue           = _diffuse(queue, neighbors)
        queue           = np.maximum(0.0, queue)
        q_history[t]    = queue
        e_history[t]    = efficiency

    return q_history, e_history