"""
qubo.py -- Layer 3: turn (part of) the construction schedule into a QUBO.

Formulation: time-indexed (Pritsker et al., 1969), the formulation chosen as
the most qubit-efficient for RCPSP on quantum annealers by Perez Armas,
Creemers & Deleplanque (Scientific Reports 14:16784, 2024).

BINARY VARIABLES
    x[j,s] = 1  if activity j starts on day s   (s ranges over a short window)
    end[t] = 1  if the whole (sub)project ends on day t   (makespan mode only)
    y[...]     slack bits, only where a resource could be over-used

ENERGY  =  OBJECTIVE  +  P * (P1 + P2 + P3)

    OBJECTIVE   makespan mode : sum_t t * end[t]          (finish as early as possible)
                start mode    : sum_j sum_s (s - lo_j) * x[j,s]   (start each activity early)

    P1 one start per activity       (sum_s x[j,s] - 1)^2
    P2 precedence  i -> j           x[i,s] * x[j,s']   for every s' < s + delta_ij
                                    (delta = minimum start gap; see project.py)
    P3 resource k on day t          usage(t,k) = sum r_jk * x[j,s] over starts s
                                    that keep j running on day t
        - capacity 1 (crane, excavator): pairwise penalty  x * x'  for any two
          different activities running on the same day -- exact, needs NO slack
        - capacity > 1: inequality usage <= cap becomes the equality
          (usage + z - cap)^2 with an integer slack z in [0, cap] written in
          binary.  Constraints that can never be violated are skipped.

PENALTY WEIGHT P
    Every violation adds at least P (all coefficients are integers, so a
    violated squared term is >= 1). Choosing P = (max objective - min
    objective) + 1 guarantees that no violation can ever lower the energy
    below the best valid schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import dimod
import numpy as np

from .project import Project

END = "__END__"


def slack_coefficients(cap: int) -> List[int]:
    """Bounded binary encoding: integer coefficients whose subset sums cover
    exactly 0..cap (e.g. cap 8 -> [1, 2, 4, 1], cap 3 -> [1, 2])."""
    m = int(math.floor(math.log2(cap))) + 1
    coeffs = [2 ** b for b in range(m - 1)]
    coeffs.append(cap - (2 ** (m - 1) - 1))
    return coeffs


def encode_slack(value: int, coeffs: List[int]) -> List[int]:
    """Bits for `value` under `slack_coefficients` (inverse of the encoding)."""
    bits = [0] * len(coeffs)
    if value > sum(coeffs[:-1]):
        bits[-1] = 1
        value -= coeffs[-1]
    for b in range(len(coeffs) - 2, -1, -1):
        if value >= coeffs[b]:
            bits[b] = 1
            value -= coeffs[b]
    assert value == 0
    return bits


@dataclass
class Qubo:
    bqm: dimod.BinaryQuadraticModel
    blocks: Dict[str, Dict[int, int]]          # activity (or END) -> {start: var}
    slack: List[Tuple[str, int, List[int], List[int], Dict[int, int]]] = field(default_factory=list)
    penalty: float = 0.0
    n_pairwise: int = 0

    @property
    def num_vars(self) -> int:
        return self.bqm.num_variables

    @property
    def num_slack(self) -> int:
        return sum(len(s[2]) for s in self.slack)

    # ------------------------------------------------------------------
    def decode(self, sample: Dict[int, int]) -> Tuple[Dict[str, int], List[str]]:
        """bitstring -> schedule. Returns (schedule, activities whose
        one-hot block was not exactly one 1)."""
        schedule, bad = {}, []
        for aid, starts in self.blocks.items():
            on = [s for s, v in starts.items() if sample.get(v, 0) == 1]
            if len(on) == 1:
                schedule[aid] = on[0]
            else:
                bad.append(aid)
        return schedule, bad

    def encode(self, schedule: Dict[str, int], project: Project) -> Dict[int, int]:
        """schedule -> bitstring, choosing the slack bits that zero the
        resource penalties (used to check the energy of known schedules)."""
        sample = {v: 0 for v in self.bqm.variables}
        for aid, starts in self.blocks.items():
            if aid in schedule and schedule[aid] in starts:
                sample[starts[schedule[aid]]] = 1
        for r, t, coeffs, vars_, terms in self.slack:
            usage = sum(c for v, c in terms.items() if sample[v])
            residual = self._cap[(r, t)] - usage
            if 0 <= residual:
                for v, b in zip(vars_, encode_slack(residual, coeffs)):
                    sample[v] = b
        return sample


def _add_square(bqm, terms: Dict[int, float], const: float, weight: float) -> None:
    """Add weight * (sum_i a_i v_i + const)^2 to the BQM (binary: v^2 = v)."""
    items = list(terms.items())
    for i, (u, a) in enumerate(items):
        bqm.add_linear(u, weight * (a * a + 2 * const * a))
        for v, b in items[i + 1:]:
            bqm.add_quadratic(u, v, weight * 2 * a * b)
    bqm.offset += weight * const * const


def build_qubo(project: Project,
               windows: Dict[str, List[int]],
               residual: Dict[str, np.ndarray],
               fixed_pred_finish: Optional[Dict[str, int]] = None,
               mode: str = "makespan",
               end_window: Optional[List[int]] = None,
               weights: Optional[Dict[str, float]] = None,
               penalty: Optional[float] = None) -> Qubo:
    """Build the QUBO for the activities in `windows`.

    windows    activity -> candidate start days (precedence with activities
               OUTSIDE the QUBO must already be respected by these windows)
    residual   resource -> array of capacity still free on each day
    mode       "makespan" (adds the END block) or "start"
    weights    start mode only: cost per day of delay for each activity
               (the hybrid uses each activity's remaining critical-path length)
    """
    acts = project.activities
    free = list(windows)
    bqm = dimod.BinaryQuadraticModel("BINARY")
    blocks: Dict[str, Dict[int, int]] = {}
    nxt = 0

    def new_var():
        nonlocal nxt
        nxt += 1
        bqm.add_variable(nxt - 1, 0.0)
        return nxt - 1

    for aid in free:
        blocks[aid] = {s: new_var() for s in windows[aid]}
    if mode == "makespan":
        blocks[END] = {t: new_var() for t in end_window}

    # ---- objective coefficients and penalty weight ---------------------
    obj = {}
    if mode == "makespan":
        # shifted so the cheapest choice costs 0: then switching the whole
        # block off can save at most obj_range, which P is guaranteed to exceed
        t0 = min(end_window)
        for t, v in blocks[END].items():
            obj[v] = float(t - t0)
        obj_range = max(end_window) - t0
    else:
        obj_range = 0
        for aid in free:
            w = (weights or {}).get(aid, 1)
            lo = min(windows[aid])
            for s, v in blocks[aid].items():
                obj[v] = float(w * (s - lo))
            obj_range += w * (max(windows[aid]) - lo)
    P = float(penalty) if penalty is not None else float(obj_range + 1)
    for v, c in obj.items():
        bqm.add_linear(v, c)

    # ---- P1: exactly one start per block --------------------------------
    for aid, starts in blocks.items():
        _add_square(bqm, {v: 1.0 for v in starts.values()}, -1.0, P)

    # ---- P2: precedence among the free activities (and to END) ----------
    edges = [(p, s, dl) for p, s, dl in project.edges() if p in blocks and s in blocks]
    if mode == "makespan":
        edges += [(a, END, acts[a].duration) for a in free]
    for p, q, dl in edges:
        for sp, vp in blocks[p].items():
            for sq, vq in blocks[q].items():
                if sq < sp + dl:
                    bqm.add_quadratic(vp, vq, P)

    # ---- P3: resource capacity -------------------------------------------
    slack = []
    cap_lookup = {}
    n_pairwise = 0
    horizon = max(max(w) + acts[a].duration for a, w in windows.items())
    for r in project.resources:
        users = [a for a in free if acts[a].demand.get(r, 0) > 0]
        if len(users) < 2:
            continue                       # windows already respect residual
        for t in range(horizon):
            terms = {}
            max_use = 0
            for a in users:
                d, u = acts[a].duration, acts[a].demand[r]
                covering = [v for s, v in blocks[a].items() if s <= t < s + d]
                for v in covering:
                    terms[v] = u
                if covering:
                    max_use += u
            cap = int(residual[r][t])
            if max_use <= cap:
                continue                   # can never be violated
            if cap == 1:                   # pairwise, no slack needed
                by_act = {}
                for a in users:
                    d = acts[a].duration
                    by_act[a] = [v for s, v in blocks[a].items() if s <= t < s + d]
                names = [a for a in users if by_act[a]]
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        for va in by_act[a]:
                            for vb in by_act[b]:
                                bqm.add_quadratic(va, vb, P)
                                n_pairwise += 1
            elif cap <= 0:                 # nothing left: every use is a violation
                for v, u in terms.items():
                    bqm.add_linear(v, P * u)
            else:
                coeffs = slack_coefficients(cap)
                svars = [new_var() for _ in coeffs]
                full = dict(terms)
                for v, c in zip(svars, coeffs):
                    full[v] = c
                _add_square(bqm, full, -float(cap), P)
                slack.append((r, t, coeffs, svars, terms))
                cap_lookup[(r, t)] = cap

    q = Qubo(bqm, blocks, slack, P, n_pairwise)
    q._cap = cap_lookup
    return q


def full_project_size(project: Project, horizon: int) -> Dict[str, int]:
    """Size of the QUBO for the WHOLE project in one piece (counted, not
    built -- building it is exactly what does not scale)."""
    acts = project.activities
    es, ls = project.earliest_starts(), project.latest_starts(horizon)
    win = {a: range(es[a], ls[a] + 1) for a in acts}
    n_x = sum(len(w) for w in win.values()) + (horizon - project.critical_path_length() + 1)
    n_slack = 0
    for r, cap in project.capacity.items():
        users = [a for a in acts if acts[a].demand.get(r, 0) > 0]
        for t in range(horizon):
            max_use = sum(acts[a].demand[r] for a in users
                          if any(s <= t < s + acts[a].duration for s in win[a]))
            if max_use > cap and cap > 1:
                n_slack += len(slack_coefficients(cap))
    return {"start_variables": n_x, "slack_variables": n_slack,
            "total_qubits": n_x + n_slack}
