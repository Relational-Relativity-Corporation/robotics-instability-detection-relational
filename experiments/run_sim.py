"""
Phase 1 — Simulation verification only.

Checks:
1. Flat baseline pre-disruption (delta Q ~ 0)
2. Cascade builds post-disruption
3. Failure threshold crossed within T
4. Lead time window exists (cascade onset visible before failure)

No detection logic. Detection layer added in Phase 2 after this is confirmed.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rb_rel.network     import build_fleet, MAX_QUEUE_DEPTH, N
from rb_rel.disruption  import gradual_degradation
from rb_rel.simulation  import simulate

T             = 300
T_DISRUPTION  = 100
DISRUPTION_BOT = 2


def main():
    params, neighbors = build_fleet()
    disruption        = gradual_degradation(DISRUPTION_BOT, T_DISRUPTION)
    q_hist, e_hist    = simulate(params, neighbors, T=T, disruption_fn=disruption)

    max_q      = q_hist.max(axis=1)
    failure_t  = next((i for i, v in enumerate(max_q) if v >= MAX_QUEUE_DEPTH), None)

    # ── Baseline check ────────────────────────────────────────
    pre       = q_hist[:T_DISRUPTION]
    pre_drift = pre[-1].mean() - pre[0].mean()
    print(f"Pre-disruption mean queue drift:   {pre_drift:.4f}  (target: ~0.0)")

    # ── Cascade check ─────────────────────────────────────────
    post      = q_hist[T_DISRUPTION:]
    post_rise = post[-1].max() - post[0].max()
    print(f"Post-disruption max queue rise:    {post_rise:.2f}")

    # ── Efficiency at disrupted robot ─────────────────────────
    print(f"R{DISRUPTION_BOT} efficiency at t=end:       {e_hist[-1, DISRUPTION_BOT]:.3f}")

    # ── Failure ───────────────────────────────────────────────
    if failure_t is not None:
        print(f"Failure threshold crossed:         t = {failure_t}")
    else:
        print(f"Failure threshold not crossed within T={T}")

    # ── Plot ──────────────────────────────────────────────────
    t_axis = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for i in range(N):
        ax1.plot(t_axis, q_hist[:, i], lw=1.0, alpha=0.7, label=f'R{i}')
    ax1.axvline(T_DISRUPTION, color='crimson', ls='--', alpha=0.7,
                label=f'Degradation onset (t={T_DISRUPTION})')
    if failure_t is not None:
        ax1.axvline(failure_t, color='darkorange', ls=':', lw=2,
                    label=f'Failure (t={failure_t})')
    ax1.axhline(MAX_QUEUE_DEPTH, color='darkorange', ls='--', alpha=0.3,
                label='Max queue depth')
    ax1.set_ylabel('Queue Depth')
    ax1.set_title('Task-Queue Fleet Simulation — Phase 1 Verification')
    ax1.legend(fontsize=7, ncol=2)

    for i in range(N):
        ax2.plot(t_axis, e_hist[:, i], lw=1.0, alpha=0.7, label=f'R{i}')
    ax2.axvline(T_DISRUPTION, color='crimson', ls='--', alpha=0.7)
    ax2.set_ylabel('Efficiency')
    ax2.set_xlabel('Time Step')
    ax2.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig('task_queue_sim_phase1.png', dpi=120)
    plt.show()
    print("Plot saved -> task_queue_sim_phase1.png")


if __name__ == '__main__':
    main()