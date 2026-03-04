import numpy as np


def compute_trigger_times(max_queue: np.ndarray, signal: np.ndarray,
                          signal_threshold: float = 0.10,
                          overload_threshold: float = 150.0):
    """
    Compute instability detection time, failure time, and lead time.

    Parameters
    ----------
    max_queue         : per-timestep maximum queue depth across fleet
    signal            : relational momentum series
    signal_threshold  : momentum value at which instability is declared
    overload_threshold: queue depth at which failure is declared
    """
    instability_time = next(
        (i for i, v in enumerate(signal)
         if not np.isnan(v) and v >= signal_threshold),
        None
    )
    failure_time = next(
        (i for i, v in enumerate(max_queue) if v >= overload_threshold),
        None
    )
    lead_time = (
        failure_time - instability_time
        if instability_time is not None and failure_time is not None
        else None
    )
    return instability_time, failure_time, lead_time