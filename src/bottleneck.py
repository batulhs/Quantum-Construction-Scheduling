"""
bottleneck.py -- which crews and equipment cause the project's delay, and
which activities fight over them.

GAP 1  rank_bottlenecks()     Resource Bottleneck Detection Algorithm
GAP 3  conflict_analysis()    Multi-resource conflict detection and conflict core
       extract_core()         activities handed to the quantum solver

Every delay measure uses the same yardstick: the shortest schedule found by
the serial schedule-generation scheme over four standard priority rules
(classical.best_sgs). Call its makespan M(capacities).

GAP 1 -- for every resource k
    utilisation      U_k  = work_k / (capacity_k * makespan)      (warm-start schedule)
    conflict days    days on which the resource-free (critical path) schedule
                     needs more of k than exists
    isolated delay   I_k  = M(only k limited)        - critical path
                     how long k alone would delay the project
    removal gain     G_k  = M(all limited) - M(all except k limited)
                     how many days are saved if k were unlimited
    impact           max(I_k, G_k); ties broken by conflict days, then by U_k
Resources are ranked by impact; impact 0 means k never delays the project.

GAP 3 -- multi-resource conflicts
    pair synergy     S_kl = M(all) - M(all except k and l) - max(G_k, G_l)
                     S_kl > 0: k and l delay the project TOGETHER by more
                     than either does alone (a joint bottleneck)
    conflict graph   nodes = activities; an edge joins two activities that are
                     both running on an over-subscribed day of the same
                     resource in the resource-free schedule; each edge is
                     labelled with every resource the pair competes for
    multi-resource   an activity in conflict on >= 2 resources (e.g. it needs
    activity         the crane AND the crew, and both are taken)
    conflict cores   connected groups of the graph restricted to bottleneck
                     resources, scored by the impact of the resources inside
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .classical import best_sgs
from .project import Project

UNLIMITED = 10 ** 6


def _makespan(project: Project, capacity: Dict[str, int]) -> int:
    p = Project(project.name, project.activities, capacity, project.info, list(project.order))
    s = best_sgs(p)
    return max(s[a] + p.activities[a].duration for a in s)


def _es_usage(project: Project) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    es = project.earliest_starts()
    T = project.critical_path_length()
    use = {r: np.zeros(T + 1, dtype=int) for r in project.resources}
    for a in project.activities.values():
        for r, u in a.demand.items():
            use[r][es[a.id]:es[a.id] + a.duration] += u
    return es, use


# ---------------------------------------------------------------------------
# GAP 1
# ---------------------------------------------------------------------------
def rank_bottlenecks(project: Project, schedule: Dict[str, int] = None) -> pd.DataFrame:
    cpm = project.critical_path_length()
    cap = project.capacity
    base = _makespan(project, cap)
    schedule = schedule or best_sgs(project)
    C = max(schedule[a] + project.activities[a].duration for a in schedule)
    _, es_use = _es_usage(project)

    rows = []
    for k in project.resources:
        isolated = _makespan(project, {r: (c if r == k else UNLIMITED) for r, c in cap.items()}) - cpm
        gain = base - _makespan(project, {r: (UNLIMITED if r == k else c) for r, c in cap.items()})
        over = es_use[k] - cap[k]
        rows.append({
            "Resource": k,
            "Capacity": cap[k],
            "Utilisation (%)": round(100 * project.total_work(k) / (cap[k] * C), 1),
            "Peak demand (no limits)": int(es_use[k].max()),
            "Conflict days": int((over > 0).sum()),
            "Isolated delay (days)": int(isolated),
            "Removal gain (days)": int(gain),
            "Impact (days)": int(max(isolated, gain)),
        })
    df = pd.DataFrame(rows).sort_values(
        ["Impact (days)", "Conflict days", "Utilisation (%)"], ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df.attrs.update(critical_path=cpm, makespan=base)
    return df


# ---------------------------------------------------------------------------
# GAP 3
# ---------------------------------------------------------------------------
def pair_synergy(project: Project, ranking: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Joint effect of relaxing two bottleneck resources together."""
    cap = project.capacity
    base = ranking.attrs["makespan"]
    gain = dict(zip(ranking["Resource"], ranking["Removal gain (days)"]))
    cands = [r for r in ranking["Resource"] if ranking.set_index("Resource").loc[r, "Impact (days)"] > 0][:top]
    rows = []
    for k, l in combinations(cands, 2):
        both = base - _makespan(project, {r: (UNLIMITED if r in (k, l) else c) for r, c in cap.items()})
        rows.append({"Resource pair": f"{k} + {l}", "Joint gain (days)": int(both),
                     "Synergy (days)": int(both - max(gain[k], gain[l]))})
    return pd.DataFrame(rows).sort_values(["Synergy (days)", "Joint gain (days)"], ascending=False) \
        .reset_index(drop=True) if rows else pd.DataFrame(columns=["Resource pair", "Joint gain (days)", "Synergy (days)"])


def conflict_graph(project: Project, resources: List[str]) -> Dict[Tuple[str, str], Set[str]]:
    """Edges (i, j) -> set of resources i and j compete for on an
    over-subscribed day of the resource-free schedule."""
    es, use = _es_usage(project)
    edges: Dict[Tuple[str, str], Set[str]] = {}
    for k in resources:
        days = np.nonzero(use[k] > project.capacity[k])[0]
        users = [a for a in project.activities.values() if a.demand.get(k, 0) > 0]
        for t in days:
            running = sorted(a.id for a in users if es[a.id] <= t < es[a.id] + a.duration)
            for i, j in combinations(running, 2):
                edges.setdefault((i, j), set()).add(k)
    return edges


def conflict_analysis(project: Project, ranking: pd.DataFrame):
    """Conflict graph on the bottleneck resources, its multi-resource
    activities, and its connected conflict cores ranked by impact."""
    impact = dict(zip(ranking["Resource"], ranking["Impact (days)"]))
    bottlenecks = [r for r in ranking["Resource"] if impact[r] > 0]
    edges = conflict_graph(project, bottlenecks)

    per_act: Dict[str, Set[str]] = {}
    for (i, j), rs in edges.items():
        per_act.setdefault(i, set()).update(rs)
        per_act.setdefault(j, set()).update(rs)
    multi = {a: rs for a, rs in per_act.items() if len(rs) >= 2}

    # connected components (union-find)
    parent = {a: a for a in per_act}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        parent[find(i)] = find(j)
    groups: Dict[str, List[str]] = {}
    for a in per_act:
        groups.setdefault(find(a), []).append(a)

    cores = []
    for members in groups.values():
        rs = set().union(*(per_act[a] for a in members))
        cores.append({"activities": sorted(members, key=lambda a: (len(a), a)),
                      "resources": sorted(rs, key=lambda r: -impact[r]),
                      "impact": sum(impact[r] for r in rs),
                      "multi_resource_activities": sorted(a for a in members if a in multi)})
    cores.sort(key=lambda c: (-c["impact"], -len(c["multi_resource_activities"]), -len(c["activities"])))
    return {"bottlenecks": bottlenecks, "edges": edges, "multi": multi, "cores": cores}


def extract_core(project: Project, analysis: dict, qubit_budget: int, build) -> Tuple[List[str], object]:
    """Pick the activities for the quantum solver from the conflict cores.

    For each core (highest impact first) and each seed activity (most
    contested resources first, then shortest): grow the group by repeatedly
    adding the core activity that shares the most contested resources with
    it (shorter first, as it needs fewer qubits) while the QUBO fits the
    budget. Members need not overlap directly: short jobs usually clash
    through the same long jobs, so sharing a contested resource within the
    same core is enough. Of all groups seen, keep the best one that contains a real
    conflict, preferring: more resources in conflict, more activities, more
    qubits used. `build(ids)` returns (qubo, has_conflict) or None."""
    edges = analysis["edges"]
    best = None
    for core in analysis["cores"]:
        members = core["activities"]
        if len(members) < 2:
            continue
        rs_of = {a: set().union(*[rs for (i, j), rs in edges.items() if a in (i, j)]) for a in members}

        def link(a, group):
            # contested resources a shares with the group (same core), plus
            # a bonus for every direct clash with a group member
            shared = len(rs_of[a] & set().union(*(rs_of[g] for g in group)))
            direct = sum(1 for (i, j) in edges if (i == a and j in group) or (j == a and i in group))
            return 2 * shared + min(direct, 1)

        def resources_in(group):
            return set().union(*(rs_of[g] for g in group)) if len(group) > 1 else set()

        seeds = sorted(members, key=lambda a: (-len(rs_of[a]), project.activities[a].duration, a))
        for seed in seeds[:12]:
            group = [seed]
            while True:
                cands = sorted((a for a in members if a not in group and link(a, group) > 0),
                               key=lambda a: (-link(a, group), project.activities[a].duration, a))
                for a in cands:
                    built = build(group + [a])
                    if built is not None and built[0].num_vars <= qubit_budget:
                        group = group + [a]
                        q, conflict = built
                        if conflict:
                            key = (len(resources_in(group)), len(group), q.num_vars)
                            if best is None or key > best[0]:
                                best = (key, list(group), q)
                        break
                else:
                    break
        if best is not None:
            return best[1], best[2]
    raise ValueError("no conflict core fits the qubit budget")
