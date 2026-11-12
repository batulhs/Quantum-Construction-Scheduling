"""
Automated tests on the REAL DSLIB projects.   Run:  py -m pytest -v

  1. reader      DSLIB Excel files are read correctly; every link type converts right
  2. checker     the shared validator really catches broken schedules
  3. classical   CP-SAT is optimal, the heuristic is valid
  4. QUBO        encoding is mathematically correct on the real slice
  5. qubits      QUBO -> qubit operator keeps the same ordering of solutions
  6. hybrid      valid and never worse than its warm start, on every project
  7. gaps        bottleneck ranking (1), adaptive decomposition (2), conflicts (3)
"""

import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.classical import cpsat, serial_sgs
from src.dslib import find_file, load_selected, read_dslib, resource_strength, validate_project
from src.evaluate import is_feasible, makespan, validate_schedule
from src.project import link_delta
from src.qubo import END, encode_slack, full_project_size, slack_coefficients
from src.quantum import analyse_counts, anneal, bitstring_to_sample, hybrid, qubo_to_ising_operator
from src.slice_problem import QUBIT_BUDGET, build_slice, exhaustive
from src.bottleneck import conflict_analysis, rank_bottlenecks

DATA = os.path.join(ROOT, "data", "dslib")


@pytest.fixture(scope="module")
def projects():
    return [p for _, p, _ in load_selected(DATA)]


@pytest.fixture(scope="module")
def edu(projects):
    return projects[0]


@pytest.fixture(scope="module")
def slice_(edu):
    sp, q, _ = build_slice(edu, "laptop")
    return sp, q


# 1. reader -----------------------------------------------------------------
def test_projects_load_and_are_valid(projects):
    assert [p.n for p in projects] == [58, 52, 58]
    for p in projects:
        assert validate_project(p) == []


def test_known_values_from_the_excel_file(edu):
    a = edu.activities
    assert a["5"].duration == 38 and a["5"].demand == {"plumber": 1}       # "38d", "plumber [1,00 #8]"
    assert ("7", 0) in a["8"].preds                                         # "7SS"  -> gap 0
    assert ("31", a["31"].duration + 3) in a["33"].preds                    # "31FS+3d"
    assert ("46", a["46"].duration - 1) in a["48"].preds                    # "46FS-1d"
    assert edu.capacity["painter"] == 3                                     # "3 people", as recorded


@pytest.mark.parametrize("kind,expected", [("FS", 5 + 2), ("SS", 2), ("FF", 5 - 3 + 2), ("SF", 2 - 3)])
def test_link_conversion(kind, expected):
    assert link_delta(kind, d_pred=5, d_succ=3, lag=2) == expected


def test_resource_strength_never_raises_capacity():
    raw, _ = read_dslib(find_file(DATA, "C2025-06"))
    tight = resource_strength(raw, 0.5)
    for r in raw.capacity:
        rmin = max(a.demand.get(r, 0) for a in raw.activities.values())
        assert rmin <= tight.capacity[r] <= raw.capacity[r]


# 2. checker ----------------------------------------------------------------
def test_checker_catches_broken_schedules(edu):
    good = serial_sgs(edu).schedule
    bad = dict(good)
    bad["14"] = bad["13"]                       # 14 must start after 13 finishes
    assert validate_schedule(edu, bad)["precedence"] > 0
    crowd = {a: 0 for a in edu.activities}      # everything on day 0
    assert validate_schedule(edu, crowd)["resource_overload"] > 0


# 3. classical --------------------------------------------------------------
def test_cpsat_optimal_and_valid(projects):
    for p in projects:
        res, lb = cpsat(p)
        assert is_feasible(p, res.schedule)
        assert makespan(p, res.schedule) == lb
        assert lb >= p.critical_path_length()


def test_crews_really_bind_in_every_project(projects):
    for p in projects:
        _, lb = cpsat(p)
        assert lb > p.critical_path_length()    # otherwise there is nothing to optimise


def test_sgs_valid(projects):
    for p in projects:
        assert is_feasible(p, serial_sgs(p).schedule)


# 4. QUBO -------------------------------------------------------------------
@pytest.mark.parametrize("cap", range(1, 17))
def test_slack_encoding_covers_exactly_0_to_cap(cap):
    coeffs = slack_coefficients(cap)
    assert sum(coeffs) == cap
    for v in range(cap + 1):
        assert sum(c * b for c, b in zip(coeffs, encode_slack(v, coeffs))) == v


def test_ground_state_is_the_true_optimum(slice_):
    sp, q = slice_
    _, opt = cpsat(sp)
    ground, _, rand = exhaustive(sp, q, opt)
    assert is_feasible(sp, ground) and makespan(sp, ground) == opt
    assert 0 < rand["optimal_share"] < rand["feasible_share"] < 1


def test_annealing_solves_core(slice_):
    sp, q = slice_
    _, opt = cpsat(sp)
    sample, _, _ = anneal(q, num_reads=300)
    sched, bad = q.decode(sample)
    sched.pop(END, None)
    assert not bad and is_feasible(sp, sched) and makespan(sp, sched) == opt


# 5. qubits -----------------------------------------------------------------
def test_ising_operator_preserves_energy_order(slice_):
    sp, q = slice_
    n = q.num_vars
    diag = np.real(qubo_to_ising_operator(q).to_matrix(sparse=True).diagonal())
    idx = np.random.default_rng(0).choice(2 ** n, 300, replace=False)
    e = [q.bqm.energy(bitstring_to_sample(format(i, f"0{n}b"), n)) for i in idx]
    assert np.corrcoef(diag[idx], e)[0, 1] > 0.999999


def test_analyse_counts_picks_shortest_valid(slice_):
    sp, q = slice_
    n = q.num_vars
    _, opt = cpsat(sp)
    ground, _, _ = exhaustive(sp, q, opt)
    from src.classical import best_sgs
    slow = {a: s + 1 for a, s in best_sgs(sp).items()}           # valid, one day later

    def bits(sch):
        smp = q.encode({**sch, END: makespan(sp, sch)}, sp)
        return "".join(str(smp[i]) for i in reversed(range(n)))
    counts = {bits(ground): 1, bits(slow): 5}
    best, _ = analyse_counts(q, counts, sp, optimum=opt)
    assert makespan(sp, best) == opt


# 6. hybrid -----------------------------------------------------------------
def test_hybrid_valid_and_not_worse(projects):
    for p in projects:
        h = hybrid(p, passes=1)
        assert int(h.notes["Max qubits/sub-QUBO"]) <= 156
        assert is_feasible(p, h.schedule)
        assert makespan(p, h.schedule) <= makespan(p, serial_sgs(p).schedule)


def test_full_projects_too_big_for_quantum_hardware(projects):
    for p in projects:
        assert full_project_size(p, makespan(p, serial_sgs(p).schedule))["total_qubits"] > 156


# 7. research gaps ----------------------------------------------------------
def test_gap1_bottleneck_ranking(projects):
    expected_top = ["painter", "plumber", "Hoogtewerker"]
    for p, top in zip(projects, expected_top):
        rk = rank_bottlenecks(p)
        assert rk.iloc[0]["Resource"] == top
        assert rk.iloc[0]["Impact (days)"] > 0
        assert list(rk["Impact (days)"]) == sorted(rk["Impact (days)"], reverse=True)
        unlimited = {r: 10 ** 6 for r in p.capacity}            # no bottleneck -> no impact
        from src.project import Project
        free = Project(p.name, p.activities, unlimited, p.info, list(p.order))
        assert rank_bottlenecks(free)["Impact (days)"].max() == 0


def test_gap3_multi_resource_conflicts(projects):
    nut = projects[2]
    an = conflict_analysis(nut, rank_bottlenecks(nut))
    assert len(an["multi"]) >= 1                                # e.g. platform AND labourers
    for a, rs in an["multi"].items():
        assert len(rs) >= 2 and all(nut.activities[a].demand.get(r, 0) > 0 for r in rs)
    for (i, j), rs in an["edges"].items():                      # every edge is a real clash
        for r in rs:
            assert nut.activities[i].demand.get(r, 0) and nut.activities[j].demand.get(r, 0)


@pytest.mark.parametrize("size", ["laptop", "gpu"])
def test_core_fits_budget_and_contains_a_conflict(edu, size):
    from src.classical import best_sgs
    sp, q, info = build_slice(edu, size)
    assert q.num_vars <= QUBIT_BUDGET[size]
    assert makespan(sp, best_sgs(sp)) > sp.critical_path_length()   # crews really delay it
    assert info["resources"][0] == "painter"                         # the top bottleneck


def test_gap2_adaptive_respects_budget_and_matches_fixed(projects):
    for p in projects:
        rk = rank_bottlenecks(p)
        w = dict(zip(rk["Resource"], rk["Impact (days)"]))
        a = hybrid(p, decomposition="adaptive", bottleneck_weight=w, qubit_budget=100)
        f = hybrid(p, decomposition="fixed")
        assert is_feasible(p, a.schedule)
        assert int(a.notes["Max qubits/sub-QUBO"]) <= 100
        assert makespan(p, a.schedule) <= makespan(p, f.schedule)


# --------------------------------------------------------------------------- web UI
def test_web_ui_loads_without_errors():
    """The Streamlit app (app.py) starts and renders every tab with no exception."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=300).run()
    assert not at.exception
    assert len(at.tabs) == 6
