# Technical Decisions, Methodology and Results
*Hybrid quantum-classical construction scheduling on real data: bottleneck detection, multi-resource conflicts, adaptive decomposition*

## 1. Problem

The **Resource-Constrained Project Scheduling Problem (RCPSP)**: choose a start day for every activity so that

- every precedence link holds,
- on every day, each crew/equipment type is used at most up to its capacity,
- the **makespan** (finish day of the last activity) is minimal.

Real schedules use four link types with lags. Each is a lower bound on the gap between start days, so every
link is stored as one rule, $S_j \ge S_i + \delta_{ij}$:

| Link | $\delta_{ij}$ |
|---|---|
| finish-to-start + lag | $d_i + \text{lag}$ |
| start-to-start + lag | $\text{lag}$ |
| finish-to-finish + lag | $d_i - d_j + \text{lag}$ |
| start-to-finish + lag | $\text{lag} - d_j$ |

$\delta$ may be negative. The classical solver, the checker and the QUBO all use this single form.

## 2. Data and project selection

**Source.** DSLIB v3.4 (Batselier & Vanhoucke, 2015; 231 real projects exported from the ProTrack scheduling software).

**Selection funnel**, from the database's own analysis sheet:

| Filter | Projects left |
|---|---|
| all projects | 231 |
| construction sector | 163 |
| renewable resources (crews/equipment) recorded | 56 |
| 30–150 activities | 39 |
| closest to the 50–100 range set by the supervisor, parsed and solved | 17 |

**Finding.** For each of the 17, CP-SAT compared the proven-optimal makespan with the critical path (the finish
date if crews were unlimited). In **16 of 17**, the recorded crew limits never delay the project: the optimum equals
the critical path, so every method ties and there is nothing to optimise. Companies recorded availability at or
above what the plan ever needs at once.

**Decision.**

1. **C2025-05 *Education Service*** is the only one of the 17 whose recorded limits bind (critical path 47 → optimum
   54 days). It is used **exactly as recorded** and is the main project.
2. Two more projects use availability reduced by the **resource-strength** rule (Kolisch, Sprecher & Drexl, 1995,
   the rule behind the PSPLIB benchmarks):
   $$R_k = r^{\min}_k + \operatorname{round}\big(RS\,(r^{\max}_k - r^{\min}_k)\big)$$
   - $r^{\min}_k$: largest single-activity demand (below it the project is impossible).
   - $r^{\max}_k$: peak daily use in the earliest-start schedule (at or above it, the resource never binds).
   - Capacities are never raised above the recorded value. Activities, durations, links and demands are untouched.
   - **C2025-06 *Primary School*** at RS = 0.5 (60 → 80 days).
   - **C2012-02 *Nut Mixing Station*** at RS = 0.2 (23 → 26 days). Chosen because it is equipment-heavy:
     excavator, assembly crane, concrete pump, pile-drilling rig.
   - Levels were chosen as the loosest of {1.0, 0.5, 0.3, 0.2} at which limits bind.

## 3. Reading the Excel files (`src/dslib.py`)

| Field | Raw example | Conversion |
|---|---|---|
| Duration | `4d`, `1d 3h`, `6h` | working days; hours **rounded up** to whole days, since a part-day still occupies the crew |
| Links | `12FS`, `6SS`, `14FS-5d`, `20FS+1w 2d` | start-gap $\delta$ (§1); weeks use the file's days-per-week |
| Demand | `carpenter [2,00 #8]` | 2 per day (comma is a decimal mark; `#8` is not a quantity); bare name = 1 |
| Availability | `12 people`, `1 mach` | 12, 1; consumable resources (materials) skipped |

Rows without a duration are WBS headings, not activities. The calendar (hours/day, days/week) comes from the
Agenda sheet. Every non-trivial conversion is counted and printed by `run_demo.py`. In the three projects:
27 start-to-start links, 8 finish-to-start links with lags, 1 finish-to-finish link with a lag, and 9 durations
rounded up from hours.

## 4. QUBO formulation (`src/qubo.py`)

Time-indexed formulation (Pritsker et al., 1969), chosen by Pérez Armas, Creemers & Deleplanque (*Sci. Rep.*
14:16784, 2024) as the most qubit-efficient for RCPSP.

**Variables.**
- $x_{j,s}=1$ if activity $j$ starts on day $s$, with $s$ in a window from the critical path.
- $e_t=1$ if the project ends on day $t$.
- $y$: slack bits.

**Energy.** $E = \text{Obj} + P\,(P_1+P_2+P_3)$, with

$$\text{Obj}=\sum_t (t-t_{\min})e_t,\quad P_1=\sum_j\Big(\sum_s x_{j,s}-1\Big)^2,\quad P_2=\sum_{i\to j}\sum_{s'<s+\delta_{ij}}x_{i,s}x_{j,s'}$$

$$P_3=\sum_{k,t}\Big(\sum_{j,\,s\le t<s+d_j} r_{jk}x_{j,s}+z_{k,t}-R_k\Big)^2$$

- **Capacity-1 resources** use the exact pairwise form $x\,x'$ for overlapping activities, which needs no slack bits.
- **Penalty weight:** $P = (\text{objective range}) + 1$. All coefficients are integers, so any violation costs at
  least $P$ and can never beat a valid schedule.

**Why the full project cannot be one QUBO.** Counted at the final schedule's horizon:

| Project | One-piece qubits |
|---|---|
| *Education Service* | 759 |
| *Primary School* | 2,238 |
| *Nut Mixing Station* | 1,188 |

A laptop simulator handles about 25 qubits, a Colab T4 GPU about 30 (a state vector needs $2^n \times 16$ bytes), and
the free IBM computer has 156 noisy qubits. The project must therefore be decomposed, which is what Gaps 1–3 make
intelligent.

## 5. Gap 1 — Resource bottleneck detection and ranking (`src/bottleneck.py`)

**Problem it solves.** The earlier version picked the quantum slice by hand (painter jobs). The scheduler did not
know *which* resource causes the delay, or by how much.

**Yardstick.** $M(R)$ = length of the best schedule found by the serial schedule-generation scheme over four
standard priority rules (LFT, LST, MTS, GRPW; Kolisch 1996) with capacities $R$. It runs in milliseconds and is
always valid.

**Per resource $k$:**

| Measure | Definition | Meaning |
|---|---|---|
| Utilisation $U_k$ | $\text{work}_k / (R_k \cdot C_{\max})$ | how busy the crew is |
| Conflict days | days on which the resource-free (critical-path) plan needs more of $k$ than exists | where clashes happen |
| Isolated delay $I_k$ | $M(\text{only } k \text{ limited}) - \text{critical path}$ | how long $k$ alone would delay the project |
| Removal gain $G_k$ | $M(\text{all}) - M(\text{all except } k)$ | days saved if $k$ were unlimited |
| **Impact** | $\max(I_k, G_k)$ | ranking key; ties broken by conflict days, then utilisation |

Both delay measures are needed. When two resources bind together, removing either alone saves little ($G_k$ small),
but each alone delays the project ($I_k$ large).

**Output.** A ranked table and chart per project, plus the weights used by Gap 2 and the core used by Gap 3.

| Project | Top bottleneck | Impact |
|---|---|---|
| *Education Service* | painters (3 available, 7 needed at peak) | 29 days |
| *Primary School* | plumbers | 20 days |
| *Nut Mixing Station* | aerial platform, then labourers, then plant technicians | 4, 1, 1 days |

The automatic choice for *Education Service* reproduces the hand-picked painter slice of the earlier version, which
serves as a validation of the method.

## 6. Gap 3 — Multi-resource conflict detection (`src/bottleneck.py`)

**Problem it solves.** An activity can wait because *several* resources are taken at once, for example the
aerial platform *and* the labourers. Single-resource analysis misses this.

1. **Conflict graph.** Nodes are activities. An edge joins two activities that both run on an over-subscribed day
   of the same bottleneck resource in the resource-free plan. Each edge is labelled with *every* resource the pair
   competes for.
2. **Multi-resource activities.** An activity in conflict on two or more resources.
3. **Pair synergy.** $S_{kl} = M(\text{all}) - M(\text{all except } k,l) - \max(G_k, G_l)$. Positive means $k$ and
   $l$ delay the project jointly by more than either alone.
4. **Conflict cores.** Connected groups of the graph, scored by the total impact of the resources inside.
5. **Core extraction for the quantum solver** (`extract_core`):
   - Start from the activity contested on the most resources (shortest first).
   - Grow the group with activities sharing the most contested resources, while the QUBO fits the qubit budget
     (16 laptop, 26 GPU).
   - Keep the group with the most resources in conflict, then the most activities.
   - The group gets **residual capacity**: what remains of each resource on those days after activities outside the
     group take their share.

**Results.** On *Nut Mixing Station* five activities are multi-resource; for example, steel assembly needs the aerial
platform *and* labourers. There are three conflict cores. The *Education Service* and *Primary School* conflicts are
each driven by a single trade.

## 7. Gap 2 — Adaptive quantum problem decomposition (`src/quantum.py`)

**Problem it solves.** The earlier hybrid used a fixed 6-day window, 3-day step, and 6 candidate days for every
activity. Sub-problem size was uncontrolled: up to 182 qubits on *Nut Mixing Station*, more than the 156-qubit
free IBM machine.

**Adaptive scheme.**
- **Conflict detection.** An activity is *waiting* if it starts later than its links allow, i.e. it waits for a crew.
- **Freedom per activity.** Waiting activities get as many candidate days as they wait (+2, up to 10), so they can be
  pulled forward. Others get 6.
- **Budget control.** If a sub-QUBO exceeds the qubit budget (default 100), candidate days are scaled down
  (×0.7, ×0.5, ×0.35), then the window is shortened (down to 3 days), until it fits. Every sub-problem therefore fits
  the target device.
- **Multi-start.** The hybrid starts from all four priority-rule plans and keeps the best result.
- **Optional expansion** (`w_max > 6`). A window can grow while a conflict crosses its boundary. It is off by default:
  in tests, widening a window under a fixed budget forced fewer candidate days per activity and gave schedules up to
  28 days longer.

**Each sub-problem:** QUBO → simulated annealing (warm-started at the current plan) → the 12 lowest-energy distinct
arrangements are each completed greedily, checked against every rule, and accepted only if the project gets shorter
(ties broken by total finish time).

**Result** (`decomposition_fixed_vs_adaptive.csv`): same schedule lengths, with every sub-problem within budget.

| Project | Fixed: days / largest sub-problem | **Adaptive: days / largest sub-problem** |
|---|---|---|
| *Education Service* | 54 / 91 qubits | **54 / 98 qubits** |
| *Primary School* | 80 / 106 qubits | **80 / 97 qubits** |
| *Nut Mixing Station* | 27 / 182 qubits | **27 / 100 qubits** |

The cost is more, smaller sub-problems: 55–164 instead of 17–50. Each sub-problem takes milliseconds on the annealer.

## 8. The quantum core (`src/slice_problem.py`)

Extracted automatically from the top conflict core of *Education Service*:

| Activity | Duration | Resource |
|---|---|---|
| 13 site cleaning | 1 day | painter |
| 14 setting out (after 13) | 1 day | painter |
| 38 concrete works | 2 days | painter |

- **Residual capacity:** 1 painter (the other 2 are occupied by the 40-day site-wide programmes).
- **QUBO:** 16 qubits.
- **Horizon:** heuristic schedule + 1 day, so the QUBO also contains valid-but-slower schedules for the solver to reject.

**QAOA.** p = 1. Angles are tuned with COBYLA on the Qiskit Aer statevector simulator (GPU when present), then the
circuit is sampled 20,000 times.

**Results:**
- The exhaustive check over all $2^{16}$ bitstrings, simulated annealing and QAOA all give the optimal 4-day schedule.
- Valid schedules: QAOA 2.6% of samples vs 0.073% for random guessing (**35×**).
- Optimal schedules: 0.68% vs 0.018% (**37×**).

On a Colab GPU the core grows to 26 qubits. That run was prepared but not completed during development: each QAOA
optimiser step at 26 qubits takes minutes on a CPU.

## 9. Validation

| Check | How |
|---|---|
| Data | no cycles, no demand above capacity |
| Every schedule | one shared checker: missing activities, link violations, crew overload, negative starts — all zero |
| Hybrid safe | never worse than its initial plan; every rebuilt schedule re-checked |
| Gap 1 | ranking sorted by impact; with unlimited resources every impact is 0 (test) |
| Gap 2 | every adaptive sub-problem ≤ budget; adaptive schedule ≤ fixed-window schedule (test + runtime check) |
| Gap 3 | every conflict edge is a real shared demand; multi-resource activities need all listed resources (test) |
| Quantum core | fits budget, contains a real conflict; lowest-energy bitstring = optimum (exhaustive) |
| Qubit mapping | operator diagonal vs QUBO energy, correlation > 0.999999 (test) |
| Reader | values checked against cells of the Excel file (test) |

`run_demo.py` prints 22 PASS/FAIL checks. CP-SAT is used only to *verify* the optimum of the quantum core. A full
classical benchmark is available with `--benchmark`.

## 10. Scope and next steps

1. Two of the three projects use reduced availability (resource-strength rule, §2). *Education Service* is fully as
   recorded.
2. Quantum annealing runs as simulated annealing. Swapping in `DWaveSampler` is a one-line change once a D-Wave API
   token is available.
3. Real IBM hardware: the 16-qubit circuit compiles to about 275 two-qubit gates on `ibm_kingston`. The run demonstrates
   the pipeline end to end on hardware.
4. Durations given in hours are rounded up to whole days.
5. Next step (Gap 4): uncertainty-aware scheduling with duration and crew-availability scenarios.

## References

- Batselier, J. & Vanhoucke, M. (2015). Construction and evaluation framework for a real-life project database. *Int. J. Project Management* 33(3), 697–710. doi:10.1016/j.ijproman.2014.09.004
- Kolisch, R. (1996). Serial and parallel resource-constrained project scheduling methods revisited: theory and computation. *European Journal of Operational Research* 90(2), 320–333.
- Kolisch, R., Sprecher, A. & Drexl, A. (1995). Characterization and generation of a general class of resource-constrained project scheduling problems. *Management Science* 41(10), 1693–1703.
- Pritsker, A.A.B., Watters, L.J. & Wolfe, P.M. (1969). Multiproject scheduling with limited resources: a zero-one programming approach. *Management Science* 16(1), 93–108.
- Pérez Armas, L.F., Creemers, S. & Deleplanque, S. (2024). Solving the resource constrained project scheduling problem with quantum annealing. *Scientific Reports* 14, 16784.
- Farhi, E., Goldstone, J. & Gutmann, S. (2014). A quantum approximate optimization algorithm. arXiv:1411.4028.
