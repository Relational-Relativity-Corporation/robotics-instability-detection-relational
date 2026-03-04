import numpy as np

DEGRADATION_RATE = 0.005
E_MIN            = 0.2


def gradual_degradation(robot_idx: int, t_disruption: int):
    """
    Gradual efficiency degradation modelling actuator or motor wear.

    Degradation model:

        e_i(t) = 1.0                                               t < t_disruption
        e_i(t) = max(e_min, 1 - degradation_rate * (t - t_disruption))  t >= t_disruption

    Parameters
    ----------
    robot_idx    : index of robot undergoing degradation
    t_disruption : timestep at which degradation begins
    """
    def fn(efficiency: np.ndarray, t: int) -> np.ndarray:
        e = efficiency.copy()
        if t >= t_disruption:
            e[robot_idx] = max(E_MIN, 1.0 - DEGRADATION_RATE * (t - t_disruption))
        return e

    return fn