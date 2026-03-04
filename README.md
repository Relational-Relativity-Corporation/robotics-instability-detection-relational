# Robotics Instability Detection — Relational Kernel

Cascade early warning in a multi-robot task-queue fleet with instability
detection derived from the ABCRE invariant relational kernel.

## Relationship to Companion Repo

This repo is a parallel implementation of
[robotics-instability-detection-demo](https://github.com/Relational-Relativity-Corporation/robotics-instability-detection-demo).

The simulation is physically grounded in task-queue dynamics rather than
abstract load values. The detection layer replaces scipy-based statistical
methods (variance, Kendall tau) with operators derived from the ABCRE
invariant relational kernel.

## System

- N robots, each maintaining a task queue
- Centralized dispatcher routes tasks proportionally to current processing
  capacity (with epsilon stabilization to model scheduler inertia)
- Local ring diffusion redistributes queue load between neighbors each step
- One robot undergoes gradual efficiency degradation at t_disruption,
  modelling actuator or motor wear
- Cascade propagates as the degraded robot processes fewer tasks,
  dispatcher redistributes arrivals, and neighboring queues grow

## Governing Equations

```
capacity_i(t)   = processing_rate * e_i(t)
arrivals_i(t)   = (capacity_i + eps) / sum(capacity_j + eps) * GLOBAL_TASK_RATE
processed_i(t)  = min(Q_i(t), capacity_i(t))
Q_i(t+1)        = Q_i(t) + arrivals_i(t) - processed_i(t) + diffusion

e_i(t) = 1.0                                                    t < t_disruption
e_i(t) = max(e_min, 1 - degradation_rate * (t - t_disruption))  t >= t_disruption
```

Failure condition:

```
max(Q_i(t)) >= MAX_QUEUE_DEPTH
```

## Detection Layer — ABCRE Derivation

Detection operates on the fleet queue depth vector `Q(t) = [Q_0, ..., Q_N-1]`.

Operators:

```
A(x)[i]      = x[i] - mean(x)                               # gradient extraction
R(x, rho)[i] = x[i] + rho * (x[(i+1) % n] - x[(i-1) % n]) # antisymmetric circulation
C(x)[i]      = x[i] / (1 + |x[i]|)                         # bounded coherence, output in (-1,1)
```

Detection pipeline:

```
relational_spread(Q)   = ||A(Q)||^2 / n          # population variance by construction
relational_momentum(W) = mean(|C(R(A(W), rho))|) # bounded monotonic trend signal
```

Replaces `np.var` and `scipy.stats.kendalltau` respectively.
No scipy dependency. Detection logic is fully derivable from the kernel.

## Broader Implication

The A-operator variance identity means the relational kernel can replace a large
portion of descriptive statistics with operator algebra.

Population variance, trend detection, and bounded coherence all derive from A, R, and C
directly. The detection logic is not a statistical procedure applied to a simulation —
it is the same relational structure that governs the simulation, operating at the
observation layer.

Where a system is governed by invariant relational operators, its observables can be
derived from those same operators rather than imported from an external statistical
framework.

## Run

```
pip install -e .
python experiments/run_demo.py
```

## Dependencies

numpy, matplotlib