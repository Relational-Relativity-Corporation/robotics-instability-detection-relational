import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rb_rel.network      import build_fleet, MAX_QUEUE_DEPTH, N
from rb_rel.disruption   import gradual_degradation
from rb_rel.simulation   import simulate
from rb_rel.instability  import relational_spread, rolling_relational_spread, relational_momentum
from rb_rel.metrics      import compute_trigger_times

T              = 300
T_DISRUPTION   = 100
DISRUPTION_BOT = 2
SIGNAL_THRESHOLD = 0.10
ROLLING_WINDOW   = 20
MOMENTUM_WINDOW  = 30
RHO              = 0.4


def main():
    params, neighbors = build_fleet()
    disruption        = gradual_degradation(DISRUPTION_BOT, T_DISRUPTION)
    q_hist, e_hist    = simulate(params, neighbors, T=T, disruption_fn=disruption)

    max_q    = q_hist.max(axis=1)
    spread_s = np.array([relational_spread(q_hist[t]) for t in range(T)])
    momentum = relational_momentum(spread_s, window=MOMENTUM_WINDOW, rho=RHO)

    valid = momentum[~np.isnan(momentum)]
    print(f"Relational momentum — max: {valid.max():.4f}  mean: {valid.mean():.4f}")

    it, ft, lt = compute_trigger_times(
        max_q, momentum,
        signal_threshold=SIGNAL_THRESHOLD,
        overload_threshold=MAX_QUEUE_DEPTH,
    )

    sig_val = momentum[it] if it is not None else float('nan')
    print(f"Relational momentum at detection:  {sig_val:.4f}")
    print(f"Instability detected:              t = {it}")
    print(f"Failure threshold crossed:         t = {ft}")
    print(f"Lead time:                         {lt} steps")

    t_axis = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for i in range(N):
        ax1.plot(t_axis, q_hist[:, i], lw=1.0, alpha=0.6, label=f'R{i}')
    ax1.axvline(T_DISRUPTION, color='crimson', ls='--', alpha=0.7,
                label=f'Degradation onset (t={T_DISRUPTION})')
    if it is not None:
        ax1.axvline(it, color='forestgreen', ls=':', lw=2,
                    label=f'Instability detected (t={it})')
    if ft is not None:
        ax1.axvline(ft, color='darkorange', ls=':', lw=2,
                    label=f'Failure (t={ft})')
    ax1.axhline(MAX_QUEUE_DEPTH, color='darkorange', ls='--', alpha=0.3,
                label='Max queue depth')
    ax1.set_ylabel('Queue Depth')
    ax1.set_title('Robotics Instability Detection — Relational Kernel (ABRCE)')
    ax1.legend(fontsize=7, ncol=2)

    ax2.plot(t_axis, momentum, color='darkorange', lw=1.5,
             label=f'Relational momentum — mean(|C(R(A(W)))|) w={MOMENTUM_WINDOW}')
    ax2.axhline(SIGNAL_THRESHOLD, color='forestgreen', ls='--', alpha=0.5,
                label=f'Detection threshold ({SIGNAL_THRESHOLD})')
    if it is not None:
        ax2.axvline(it, color='forestgreen', ls=':', lw=2)
    ax2.set_ylabel('Relational Momentum')
    ax2.set_xlabel('Time Step')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('robotics_instability_detection_relational.png', dpi=120)
    plt.show()
    print("Plot saved -> robotics_instability_detection_relational.png")


if __name__ == '__main__':
    main()
