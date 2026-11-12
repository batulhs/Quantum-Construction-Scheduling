"""
quantum.py -- Layer 4: quantum and hybrid solvers.

anneal()          Simulated annealing on a QUBO (dwave-samplers). This is the
                  classical stand-in for a D-Wave quantum annealer; on real
                  hardware the same BQM would be sent to DWaveSampler.

qaoa()            QAOA (Quantum Approximate Optimisation Algorithm).
                  1. QUBO -> Ising Hamiltonian (Pauli Z / ZZ terms)
                  2. QAOA circuit with p layers (QAOAAnsatz)
                  3. classical optimiser (COBYLA) tunes the angles using an
                     exact statevector simulator on the laptop
                  4. the tuned circuit is SAMPLED on any sampler:
                       - laptop statevector simulator  (default)
                       - noisy local replica of IBM hardware (FakeKingston)
                       - real IBM quantum computer      (run_ibm.py)
                  5. every sampled bitstring is decoded; the lowest-energy
                     sample is reported, plus the share of feasible and
                     optimal samples.

hybrid()          THE PROPOSED SYSTEM -- quantum-classical decomposition.
                  The 60-activity project is far too large for one QUBO, so:
                  * classical: order activities by priority (LFT rule) and cut
                    the list into chunks of k activities;
                  * quantum:   for each chunk, build a small QUBO (others fixed,
                    their resource use subtracted from capacity) and solve it
                    with annealing;
                  * classical: decode, check every activity, repair any that
                    break a constraint (counted and reported).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from .classical import earliest_feasible_start, new_usage, place, priority_order
from .evaluate import SolverResult
from .project import Project
from .qubo import END, Qubo, build_qubo


# ----------------------------------------------------------------------
# Simulated annealing
# ----------------------------------------------------------------------
def anneal(qubo: Qubo, num_reads: int = 200, seed: int = 1):
    ss = SimulatedAnnealingSampler().sample(qubo.bqm, num_reads=num_reads, seed=seed)
    best = ss.first
    return {int(k): int(v) for k, v in best.sample.items()}, float(best.energy), ss


# ----------------------------------------------------------------------
# QAOA
# ----------------------------------------------------------------------
def qubo_to_ising_operator(qubo: Qubo):
    """BINARY QUBO -> Pauli operator.  x = 1 maps to qubit state |1>,
    i.e. x = (1 - Z) / 2. Returned operator is normalised (max |coef| = 1);
    scaling does not change which bitstring has the lowest energy."""
    from qiskit.quantum_info import SparsePauliOp

    n = qubo.num_vars
    h, J, _ = qubo.bqm.to_ising()        # spins s = 2x - 1 = -Z
    terms = [("Z", [i], -c) for i, c in h.items() if c != 0]
    terms += [("ZZ", [i, j], c) for (i, j), c in J.items() if c != 0]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()
    return op / float(np.max(np.abs(op.coeffs)))


def bitstring_to_sample(bits: str, n: int) -> Dict[int, int]:
    """Qiskit bitstrings are little-endian: qubit 0 is the RIGHTMOST char."""
    return {i: int(bits[n - 1 - i]) for i in range(n)}


def aer_device() -> str:
    """'GPU' when qiskit-aer-gpu and an NVIDIA GPU are present (e.g. Google
    Colab), otherwise 'CPU'. Same results either way; the GPU is faster for
    large circuits (~20+ qubits)."""
    try:
        from qiskit_aer import AerSimulator
        return "GPU" if "GPU" in AerSimulator().available_devices() else "CPU"
    except Exception:
        return "CPU"


def optimise_qaoa(qubo: Qubo, reps: int = 1, maxiter: int = 80, restarts: int = 2, seed: int = 7):
    """Tune QAOA angles on the laptop with Qiskit Aer's exact (noise-free)
    statevector simulator -- compiled C++, ~0.1 s per energy evaluation."""
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    from scipy.optimize import minimize

    op = qubo_to_ising_operator(qubo)
    ansatz = QAOAAnsatz(op, reps=reps)
    dev = aer_device()
    compiled = transpile(ansatz, AerSimulator(method="statevector", device=dev), optimization_level=1)
    est = AerEstimator(options={"backend_options": {"method": "statevector", "device": dev},
                                "default_precision": 0})        # exact expectation
    rng = np.random.default_rng(seed)

    def cost(theta):
        return float(est.run([(compiled, op, theta)]).result()[0].data.evs)

    # start 1: "linear ramp" (gamma grows, beta shrinks layer by layer, like a
    # discretised adiabatic path); later starts: random angles
    ramp_g = [0.75 * (k + 0.5) / reps for k in range(reps)]
    ramp_b = [0.75 * (1 - (k + 0.5) / reps) for k in range(reps)]
    names = [p.name for p in ansatz.parameters]           # beta[..] then gamma[..]
    ramp = np.array([ramp_b[int(n[n.index("[") + 1:-1])] if n.startswith("β") else
                     ramp_g[int(n[n.index("[") + 1:-1])] for n in names])
    best = None
    for k in range(restarts):
        x0 = ramp if k == 0 else rng.uniform(0, np.pi, ansatz.num_parameters)
        res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return ansatz, best.x, float(best.fun)


def sample_qaoa(qubo: Qubo, ansatz, theta, sampler=None, backend=None, shots: int = 4000):
    """Sample the tuned circuit.
    no backend          -> laptop, noise-free Aer simulator
    backend + sampler   -> IBM hardware (real or the FakeKingston replica);
                           the circuit is compiled to that chip's native gates
                           and qubit wiring first."""
    from qiskit.transpiler import generate_preset_pass_manager

    circ = ansatz.assign_parameters(theta)
    circ.measure_all()
    if backend is None:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        dev = aer_device()
        backend = AerSimulator(method="statevector", device=dev)
        sampler = sampler or AerSampler(seed=11, options={"backend_options": {"device": dev}})
    circ = generate_preset_pass_manager(backend=backend, optimization_level=3).run(circ)
    result = sampler.run([circ], shots=shots).result()[0]
    return result.data.meas.get_counts(), circ


def analyse_counts(qubo: Qubo, counts: Dict[str, int], project: Project, optimum: Optional[int] = None):
    """Decode every sampled bitstring; return best schedule + statistics."""
    from .evaluate import is_feasible, makespan

    n = qubo.num_vars
    total = sum(counts.values())
    best_e, best_sched = None, None
    feas = opt = 0
    makespans = {}
    for bits, c in counts.items():
        sample = bitstring_to_sample(bits, n)
        sched, bad = qubo.decode(sample)
        sched.pop(END, None)
        ok = not bad and len(sched) == project.n and is_feasible(project, sched)
        if ok:
            feas += c
            ms = makespan(project, sched)
            makespans[ms] = makespans.get(ms, 0) + c
            if optimum is not None and ms == optimum:
                opt += c
        if ok:
            # best = shortest true makespan (shared evaluator), ties -> energy
            key = (makespan(project, sched), qubo.bqm.energy(sample))
            if best_e is None or key < best_e:
                best_e, best_sched = key, sched
    return best_sched, {"feasible_share": feas / total,
                        "optimal_share": opt / total if optimum is not None else None,
                        "makespans": makespans, "shots": total}


def qaoa(qubo: Qubo, project: Project, reps: int = 1, maxiter: int = 80,
         optimum: Optional[int] = None, shots: int = 20000) -> Tuple[SolverResult, dict]:
    t0 = time.perf_counter()
    ansatz, theta, _ = optimise_qaoa(qubo, reps=reps, maxiter=maxiter)
    counts, _ = sample_qaoa(qubo, ansatz, theta, shots=shots)
    sched, stats = analyse_counts(qubo, counts, project, optimum)
    stats["theta"] = theta
    stats["ansatz"] = ansatz
    res = SolverResult(f"QAOA p={reps} ({aer_device()} simulator)", sched or {}, time.perf_counter() - t0,
                       {"Qubits": str(qubo.num_vars),
                        "Feasible shots (%)": f"{100 * stats['feasible_share']:.3f}",
                        "Optimal shots (%)": f"{100 * stats['optimal_share']:.4f}" if optimum else "-"})
    return res, stats


# ----------------------------------------------------------------------
# Hybrid quantum-classical method (the proposed system)
# ----------------------------------------------------------------------
def _complete(project: Project, order: List[str], fixed: Dict[str, int]):
    """Keep `fixed` starts, place every other activity greedily (SGS).
    Returns None if `fixed` itself breaks a rule."""
    from .evaluate import is_feasible
    sched, usage = {}, new_usage(project)
    for aid in sorted(fixed, key=lambda a: fixed[a]):
        place(project, aid, fixed[aid], sched, usage)
    if not is_feasible(project.subproject(list(fixed), "part"), fixed):
        return None
    for aid in order:
        if aid not in sched:
            if any(p not in sched for p, _ in project.activities[aid].preds):
                return None
            place(project, aid, earliest_feasible_start(project, aid, sched, usage), sched, usage)
    # real projects have start-to-start and negative-lag links, so a re-built
    # schedule is only accepted if it passes the full shared checker
    return sched if is_feasible(project, sched) else None


def _windows(project, free, fixed, current, residual, H, lookahead):
    """Candidate start days for each free activity: the first `lookahead[a]`
    days that respect links from fixed activities and fit the residual
    capacity, plus its current day (so the current plan is always one valid
    answer of the sub-QUBO)."""
    acts = project.activities
    windows = {}
    for aid in sorted(free, key=lambda x: current[x]):
        a = acts[aid]
        lo = max([0] + [fixed[p] + dl for p, dl in a.preds if p in fixed])
        lo = max([lo] + [min(windows[p]) + dl for p, dl in a.preds if p in windows])
        cand, s = [], lo
        while len(cand) < lookahead[aid] and s + a.duration < H:
            if all(np.all(residual[r][s:s + a.duration] >= u) for r, u in a.demand.items()):
                cand.append(s)
            s += 1
        windows[aid] = sorted(set(cand) | {current[aid]})
    return windows


def _residual(project, fixed, cap):
    usage = new_usage(project)
    for aid, s in fixed.items():
        for r, u in project.activities[aid].demand.items():
            usage[r][s:s + project.activities[aid].duration] += u
    return {r: cap[r] - usage[r] for r in project.resources}


def fixed_subproblems(project, current, window_days=6, step=3, lookahead=6, **_):
    """Baseline decomposition: a fixed 6-day window sliding 3 days at a time,
    6 candidate days for every activity."""
    acts = project.activities
    C = max(current[a] + acts[a].duration for a in acts)
    for a0 in range(0, C, step):
        free = [a for a in acts if a0 <= current[a] < a0 + window_days]
        if len(free) >= 2:
            yield a0, free, {a: lookahead for a in free}


def adaptive_subproblems(project, current, qubit_budget=100, weight=None, H=None, cap=None,
                         step=3, w_base=6, w_min=3, w_max=6, **_):
    """GAP 2 -- adaptive decomposition. The plan is still swept every `step`
    days, but each sub-problem is sized to the conflict it contains and to
    the qubit budget, instead of a fixed 6-day window with 6 candidate days
    for every activity:

    * CONFLICT DETECTION: an activity is "waiting" if it starts later than
      its links allow -- it waits for a crew or equipment.
    * EXPAND (optional, w_max > 6; off by default because on these projects
      it did not shorten the schedule): the window grows while a waiting activity
      inside it is blocked by an activity that starts just after it -- on ANY
      resource it needs, crew or equipment (GAP 3) -- as long as the larger
      sub-problem still fits the budget with full freedom (tests showed that
      buying a wider window with fewer candidate days per activity hurts).
    * FREEDOM PER ACTIVITY: 6 candidate days each; waiting activities get as
      many as they wait (+2, up to 10) so they can be pulled forward.
    * SHRINK TO FIT: if the sub-QUBO exceeds the qubit budget, candidate days
      are scaled down, then the window is shortened (down to 3 days), until
      it fits. Every sub-problem therefore fits the target quantum device.
    """
    acts = project.activities
    weight = weight or {}
    C = max(current[a] + acts[a].duration for a in acts)
    pe = {a: max([0] + [current[p] + dl for p, dl in acts[a].preds]) for a in acts}
    waiting = {a for a in acts if current[a] > pe[a]}

    def blockers(a):
        need = set(acts[a].demand)
        return [b for b in acts if b != a and need & set(acts[b].demand)
                and current[b] < current[a] + acts[a].duration and current[b] + acts[b].duration > pe[a]]

    def fit(a0, free, scales=(1.0, 0.7, 0.5, 0.35)):
        fixed = {x: s for x, s in current.items() if s < a0}
        residual = _residual(project, fixed, cap)
        for scale in scales:
            look = {}
            for x in free:
                if x in waiting:
                    base = max(6, min(current[x] - pe[x], 8) + 2)
                else:
                    base = 6
                look[x] = max(2, int(round(scale * base)))
            windows = _windows(project, free, fixed, current, residual, H, look)
            if build_qubo(project, windows, residual, mode="start").num_vars <= qubit_budget:
                return look
        return None

    for a0 in range(0, C, step):
        w = w_base
        while w < w_max:                   # expand while a conflict crosses the boundary,
            inside = [x for x in acts if a0 <= current[x] < a0 + w and x in waiting]
            if not any(a0 + w <= current[b] < a0 + w_max for x in inside for b in blockers(x)):
                break
            bigger = [x for x in acts if a0 <= current[x] < a0 + w + 1]
            if fit(a0, bigger, scales=(1.0,)) is None:
                break                      # ...but only if it fits WITHOUT losing freedom
            w += 1
        while w >= w_min:                  # shrink until it fits the budget
            free = [x for x in acts if a0 <= current[x] < a0 + w]
            if len(free) < 2:
                break
            look = fit(a0, free)
            if look is not None:
                yield a0, free, look
                break
            w -= 1


def hybrid(project: Project, decomposition: str = "adaptive", qubit_budget: int = 100,
           passes: int = 3, num_reads: int = 60, candidates: int = 12, seed: int = 1,
           bottleneck_weight: Optional[Dict[str, float]] = None, **fixed_kw) -> SolverResult:
    """THE PROPOSED SYSTEM.
    1. classical warm start (best of four priority rules)
    2. decompose the plan into small sub-problems (adaptive by default)
    3. each sub-problem -> QUBO -> annealing returns many low-energy arrangements
    4. classical check: rebuild the rest of the plan around each arrangement,
       keep it only if it passes every rule and shortens the project."""
    from .classical import best_sgs
    from .evaluate import makespan

    t0 = time.perf_counter()
    acts = project.activities
    order = priority_order(project)
    from .classical import PRIORITY_RULES, serial_sgs
    starts = [serial_sgs(project, r).schedule for r in PRIORITY_RULES]  # 1. warm starts
    start_ms = min(makespan(project, s) for s in starts)
    if decomposition != "adaptive":
        starts = [serial_sgs(project, "LFT").schedule]       # fixed baseline: one start

    def score(sch):
        # primary: makespan; tie-break: sum of finish days (lets the search
        # accept moves that free resources without yet shortening the project)
        return (makespan(project, sch), sum(sch[a] + acts[a].duration for a in sch))
    H = len(next(iter(new_usage(project).values())))
    cap = {r: np.full(H, c) for r, c in project.capacity.items()}

    succ = project.successors()
    tail = {}
    for a in reversed(project.order):
        tail[a] = max([acts[a].duration] + [dl + tail[x] for x, dl in succ[a]])

    gen = adaptive_subproblems if decomposition == "adaptive" else fixed_subproblems
    sizes, accepted, overall = [], 0, None
    for current in starts:
        best = score(current)
        for _ in range(passes):
            improved = False
            for a0, free, look in list(gen(project, current, qubit_budget=qubit_budget,
                                           weight=bottleneck_weight, H=H, cap=cap, **fixed_kw)):
                fixed = {a: s for a, s in current.items() if s < a0}
                residual = _residual(project, fixed, cap)
                windows = _windows(project, free, fixed, current, residual, H, look)
                q = build_qubo(project, windows, residual, mode="start", weights={a: tail[a] for a in free})
                while decomposition == "adaptive" and q.num_vars > qubit_budget and max(look.values()) > 2:
                    # the plan changed since this sub-problem was sized: shrink to fit
                    look = {a: max(2, int(l * 0.7)) for a, l in look.items()}
                    windows = _windows(project, free, fixed, current, residual, H, look)
                    q = build_qubo(project, windows, residual, mode="start",
                                   weights={a: tail[a] for a in free})
                sizes.append(q.num_vars)
                warm = q.encode({a: current[a] for a in free}, project)
                ss = SimulatedAnnealingSampler().sample(q.bqm, num_reads=num_reads,
                                                        seed=seed + len(sizes), initial_states=[warm])
                seen = set()
                for rec in ss.data(["sample"], sorted_by="energy"):
                    proposal, bad = q.decode({int(k): int(v) for k, v in rec.sample.items()})
                    key = tuple(sorted(proposal.items()))
                    if bad or key in seen:
                        continue
                    seen.add(key)
                    full = _complete(project, order, {**fixed, **proposal})
                    if full is not None and score(full) < best:
                        current, best, improved = full, score(full), True
                        accepted += 1
                    if len(seen) >= candidates:
                        break
            if not improved:
                break
        if overall is None or best < overall[0]:
            overall = (best, current)
    current = overall[1]

    label = "adaptive" if decomposition == "adaptive" else "fixed window"
    return SolverResult(f"Hybrid quantum-classical ({label})", current, time.perf_counter() - t0,
                        {"Warm-start makespan": str(start_ms),
                         "Sub-QUBOs solved": str(len(sizes)),
                         "Improvements accepted": str(accepted),
                         "Max qubits/sub-QUBO": str(max(sizes, default=0)),
                         "Mean qubits/sub-QUBO": f"{np.mean(sizes):.0f}" if sizes else "0"})
