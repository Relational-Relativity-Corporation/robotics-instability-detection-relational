import numpy as np

GLOBAL_TASK_RATE = 30.0
ALPHA            = 0.15
DRAIN_RATE       = 4.5   # slightly below inflow — not equilibrium in the physical sense,
                          # but a low-gradient operating regime within D.
                          # disruption amplifies relational gradients toward the domain boundary.


def _distribute(load: np.ndarray, efficiency: np.ndarray,
                max_capacity: np.ndarray) -> np.ndarray:
    eff_cap = max_capacity * efficiency
    share   = (eff_cap / eff_cap.sum()) * GLOBAL_TASK_RATE
    return load + share


# System balance model
#
# The load dynamics follow:
#
#     L(t+1) = L(t) + inflow - drain + diffusion
#
# GLOBAL_TASK_RATE introduces work into the network each step.
# Each robot drains work proportional to its efficiency.
#
# DRAIN_RATE is chosen such that:
#
#     total inflow ~= total drain
#
# creating a low-gradient operating regime. When efficiency drops
# (disruption event), drain capacity decreases and load redistributes
# across the network, generating the cascade detected by the
# relational kernel.
def _drain(load: np.ndarray, efficiency: np.ndarray) -> np.ndarray:
    """Each robot consumes load proportional to its current efficiency."""
    return np.maximum(0.0, load - efficiency * DRAIN_RATE)


def _diffuse(load: np.ndarray, neighbors: list) -> np.ndarray:
    new_load = load.copy()
    for i, nbrs in enumerate(neighbors):
        delta = sum(load[j] - load[i] for j in nbrs)
        new_load[i] += ALPHA * delta
    return new_load


def simulate(params: dict, neighbors: list, T: int = 300,
             disruption_fn=None) -> np.ndarray:
    load       = params['current_load'].copy()
    efficiency = params['efficiency'].copy()
    max_cap    = params['max_capacity'].copy()
    history    = np.zeros((T, len(load)))

    for t in range(T):
        if disruption_fn is not None:
            efficiency = disruption_fn(efficiency, t)
        load       = _distribute(load, efficiency, max_cap)
        load       = _drain(load, efficiency)
        load       = _diffuse(load, neighbors)
        history[t] = load

    return history