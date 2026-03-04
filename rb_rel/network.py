import numpy as np

N                = 6
GLOBAL_TASK_RATE = 60.0
PROCESSING_RATE  = GLOBAL_TASK_RATE / N   # = 10.0 — exact baseline balance
MAX_QUEUE_DEPTH  = 150.0
EPS              = 0.01 * PROCESSING_RATE  # = 0.1 — dispatcher stabilization


def build_fleet():
    """
    Initialize fleet state and ring topology.

    Returns
    -------
    params : dict
        Initial queue depths and efficiency values for each robot.
    neighbors : list of list
        Ring neighbor indices for local diffusion.
    """
    params = {
        'queue':      np.zeros(N, dtype=float),
        'efficiency': np.ones(N,  dtype=float),
    }
    neighbors = [
        [(i - 1) % N, (i + 1) % N]
        for i in range(N)
    ]
    return params, neighbors