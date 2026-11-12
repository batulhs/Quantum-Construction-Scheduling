"""
app.py -- interactive web UI for the hybrid quantum-classical construction scheduler.

    streamlit run app.py

Every number on screen is computed live by the same code as run_demo.py
(src/), on the real DSLIB construction projects in data/dslib/.
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.bottleneck import conflict_analysis, pair_synergy, rank_bottlenecks
from src.classical import cpsat
from src.dslib import load_selected
from src.evaluate import is_feasible, makespan, usage_profile, utilisation, validate_schedule
from src.project import Project
from src.quantum import aer_device, anneal, hybrid
from src.qubo import END, full_project_size
from src.slice_problem import build_slice
from src.ui_jobs import run_qaoa
from src.visualize import en

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dslib")

# Colour scheme "Blueprint & Safety Amber". Mid-tone colours + see-through cards, so every box stays
# readable in BOTH light and dark mode, and switching mode needs no reload.
PRIMARY, ACCENT, DELAY, STEEL, MUTED = "#14A098", "#E3A21A", "#E06C53", "#4A90B8", "#8C9692"
EDGE = "rgba(140, 150, 146, 0.35)"
CARD = "rgba(140, 150, 146, 0.08)"
PAL = {"soft": "#BFE3DF", "card": CARD}
PHASE_PALETTE = ["#14A098", "#E3A21A", "#4A90B8", "#E06C53", "#7FB069", "#B98B5E",
                 "#5FB4D9", "#C98BA0", "#A8A86A", "#6C8FC9", "#C46A5E", "#6FB7AD"]

st.set_page_config(page_title="Quantum Construction Scheduler", page_icon="⚛️", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1350px; }}
.hero {{ position: relative; overflow: hidden; border: 1px solid {EDGE};
        background: linear-gradient(120deg, #0F2A30 0%, #0B5E5A 60%, #0B7A75 100%);
        border-radius: 18px; padding: 28px 34px 36px; margin-bottom: 18px; }}
.hero::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 7px;
        background: repeating-linear-gradient(-45deg, {ACCENT} 0 14px, #1D2B2F 14px 28px); }}
.hero h1 {{ color: #FFFFFF !important; font-size: 2.1rem; margin: 0 0 6px 0; }}
.hero p {{ color: #D7ECE9 !important; font-size: 1.05rem; margin: 0; }}
.pill {{ display:inline-block; background: rgba(255,255,255,.10); border:1px solid {ACCENT}99;
        border-radius: 999px; padding: 3px 12px; margin: 10px 6px 0 0; font-size: .85rem; color: #FFFFFF; }}
.step {{ background: {CARD}; border: 1px solid {EDGE}; border-top: 4px solid {PRIMARY};
        border-radius: 14px; padding: 14px 16px; height: 100%; color: inherit; }}
.step b {{ color: {PRIMARY}; }}
.step.q {{ border-top-color: {ACCENT}; background: {ACCENT}1F; }}
.step.q b {{ color: {ACCENT}; }}
.note {{ color: {MUTED}; font-size: .88rem; }}
div[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {EDGE}; border-left: 4px solid {PRIMARY};
        border-radius: 12px; padding: 10px 14px; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data & cached computations
@st.cache_resource(show_spinner="Loading real construction projects (DSLIB)...")
def projects():
    return {spec["code"]: (spec, p, notes) for spec, p, notes in load_selected(DATA)}


def variant(code: str, caps: tuple) -> Project:
    """The project with what-if crew sizes applied."""
    _, p, _ = projects()[code]
    if not caps:
        return p
    cap = dict(p.capacity)
    cap.update(dict(caps))
    return Project(p.name, p.activities, cap, dict(p.info), list(p.order))


@st.cache_data(show_spinner=False)
def bottlenecks(code, caps):
    rk = rank_bottlenecks(variant(code, caps))
    return rk, dict(rk.attrs)


@st.cache_data(show_spinner=False)
def conflicts(code, caps):
    p = variant(code, caps)
    rk, attrs = bottlenecks(code, caps)
    rk.attrs.update(attrs)
    return conflict_analysis(p, rk), pair_synergy(p, rk)


@st.cache_data(show_spinner=False)
def schedule(code, caps, budget):
    p = variant(code, caps)
    rk, _ = bottlenecks(code, caps)
    h = hybrid(p, decomposition="adaptive", qubit_budget=budget,
               bottleneck_weight=dict(zip(rk["Resource"], rk["Impact (days)"])))
    return h.schedule, h.notes, h.runtime_s


@st.cache_resource(show_spinner=False)
def quantum_core(code, caps, size):
    p = variant(code, caps)
    sp, q, info = build_slice(p, size)
    _, opt = cpsat(sp)
    return sp, q, info, opt


@st.cache_data(show_spinner=False)
def summary_table():
    f = os.path.join(HERE, "results", "summary_all_projects.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


# --------------------------------------------------------------------------- chart helpers
def gantt(p: Project, sched: dict, title: str, height_per_row=19):
    ids = sorted(sched, key=lambda a: (sched[a], a))
    phases = list(dict.fromkeys(p.activities[a].phase or "Project" for a in p.order))
    colour = {ph: PHASE_PALETTE[i % len(PHASE_PALETTE)] for i, ph in enumerate(phases)}
    fig = go.Figure()
    for ph in phases:
        rows = [a for a in ids if (p.activities[a].phase or "Project") == ph]
        if not rows:
            continue
        acts = [p.activities[a] for a in rows]
        fig.add_bar(
            y=[f"{a.id} · {a.name[:38]}" for a in acts], x=[max(a.duration, 0.2) for a in acts],
            base=[sched[a.id] for a in acts], orientation="h", name=ph[:30], marker_color=colour[ph],
            customdata=[[a.name, sched[a.id], sched[a.id] + a.duration, ", ".join(f"{en(r)}×{u}" for r, u in a.demand.items()) or "–"]
                        for a in acts],
            hovertemplate="<b>%{customdata[0]}</b><br>day %{customdata[1]} → %{customdata[2]}<br>%{customdata[3]}<extra></extra>")
    C = makespan(p, sched)
    fig.add_vline(x=C, line_dash="dash", line_color=DELAY, annotation_text=f"finish: day {C}",
                  annotation_position="top")
    order = [f"{a} · {p.activities[a].name[:38]}" for a in ids]
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], tickfont_size=10)
    fig.update_layout(title=title, barmode="overlay", height=max(320, height_per_row * len(ids) + 120),
                      xaxis_title="Working day", legend_title="Phase", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def resource_chart(p: Project, sched: dict, resource: str):
    prof = usage_profile(p, sched)[resource]
    days = np.arange(len(prof))
    fig = go.Figure()
    fig.add_scatter(x=days, y=prof, mode="lines", line_shape="hv", fill="tozeroy",
                    line_color=STEEL, name="in use")
    fig.add_hline(y=p.capacity[resource], line_dash="dash", line_color=DELAY,
                  annotation_text=f"available: {p.capacity[resource]}")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title=en(resource), xaxis_title="Working day", showlegend=False)
    fig.update_yaxes(rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- sidebar
P = projects()
codes = list(P)
labels = {c: f"{P[c][1].info['label']}  ({c})" for c in codes}

with st.sidebar:
    st.markdown("### ⚛️ Quantum Construction Scheduler")
    code = st.selectbox("Construction project", codes, format_func=lambda c: labels[c])
    spec, base, notes = P[code]
    st.caption(f"{base.n} activities · {len(base.edges())} links · {len(base.capacity)} crews/equipment  \n"
               f"Crew availability: {'as recorded' if spec['rs'] is None else 'reduced (resource strength ' + str(spec['rs']) + ')'}")

    st.markdown("#### What-if: crew sizes")
    rk0, _ = bottlenecks(code, ())
    top = [r for r in rk0["Resource"]][:4]
    caps = []
    for r in top:
        need = max(a.demand.get(r, 0) for a in base.activities.values())
        v = st.number_input(f"{en(r)}", min_value=int(need), max_value=int(base.capacity[r]) + 10,
                            value=int(base.capacity[r]), step=1, key=f"cap_{code}_{r}")
        if v != base.capacity[r]:
            caps.append((r, int(v)))
    caps = tuple(sorted(caps))
    if caps:
        st.info("What-if active: " + ", ".join(f"{en(r)} = {v}" for r, v in caps))

    st.markdown("#### Quantum settings")
    budget = st.slider("Qubit budget per sub-problem", 40, 156, 100, 4,
                       help="Largest QUBO the hybrid may send to the quantum solver. 156 = IBM's free ibm_kingston.")
    st.caption(f"QAOA simulator: **{aer_device()}**")
    st.caption("🌗 Light / dark mode: ⋮ menu (top right) → Settings → Theme")

project = variant(code, caps)
rk, rk_attrs = bottlenecks(code, caps)

# --------------------------------------------------------------------------- header
st.markdown(f"""
<div class="hero">
  <h1>Hybrid Quantum-Classical Construction Scheduling</h1>
  <p>Finds the crews that delay a real construction project, isolates the conflicts, and sends quantum-sized
  pieces to QAOA and quantum annealing, then rebuilds a valid, shorter schedule.</p>
  <span class="pill">Real data: DSLIB, Ghent University</span>
  <span class="pill">Gap 1 · Bottleneck ranking</span>
  <span class="pill">Gap 2 · Adaptive decomposition</span>
  <span class="pill">Gap 3 · Multi-resource conflicts</span>
  <span class="pill">QAOA · Quantum annealing</span>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "① Bottlenecks", "② Conflicts", "③ Hybrid schedule", "④ Quantum core", "📐 Method"])

# --------------------------------------------------------------------------- Overview
with tabs[0]:
    cpm = rk_attrs["critical_path"]
    k = st.columns(4)
    k[0].metric("Activities", project.n)
    k[1].metric("Critical path (no crew limits)", f"{cpm} days")
    k[2].metric("Top bottleneck", en(rk.iloc[0]["Resource"]), f"{int(rk.iloc[0]['Impact (days)'])} days of delay",
                delta_color="inverse")
    k[3].metric("One-piece QUBO size", f"{full_project_size(project, rk_attrs['makespan'])['total_qubits']:,} qubits",
                "too big for today's devices", delta_color="off")

    st.markdown("#### How it works")
    steps = [("1 · Read", "Real project: activities, links, crew & equipment needs", ""),
             ("2 · Rank", "Which crew delays the project, by how many days", ""),
             ("3 · Isolate", "Groups of jobs fighting over the same crews", ""),
             ("4 · Split", "Conflict-sized pieces that fit the qubit budget", ""),
             ("5 · Quantum", "Each piece → QUBO → annealing / QAOA", "q"),
             ("6 · Rebuild", "Keep only valid, shorter schedules", "")]
    cols = st.columns(len(steps))
    for c, (h, d, cls) in zip(cols, steps):
        c.markdown(f'<div class="step {cls}"><b>{h}</b><br><span style="font-size:.9rem">{d}</span></div>',
                   unsafe_allow_html=True)

    st.markdown("#### Results from the latest full run")
    summ = summary_table()
    if summ is not None:
        st.dataframe(summ, hide_index=True, width="stretch")
        st.caption("From `run_demo.py` (results/summary_all_projects.csv). Use the tabs to recompute any project live.")

# --------------------------------------------------------------------------- Gap 1
with tabs[1]:
    st.subheader("Which crews delay the project?")
    st.markdown(f"Critical path ignoring crews: **{rk_attrs['critical_path']} days** → with real crew limits: "
                f"**{rk_attrs['makespan']} days**. The ranking shows where those extra days come from.")
    top8 = rk.head(8).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=[en(r) for r in top8["Resource"]], x=top8["Impact (days)"], orientation="h",
                marker_color=[DELAY if v > 0 else MUTED for v in top8["Impact (days)"]],
                text=top8["Impact (days)"], textposition="outside", name="Delay caused")
    fig.update_layout(height=360, title="Delay caused by each resource (days)",
                      margin=dict(l=10, r=30, t=50, b=10), showlegend=False)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    util = px.bar(top8, x="Utilisation (%)", y=[en(r) for r in top8["Resource"]], orientation="h",
                  color="Utilisation (%)", color_continuous_scale=[[0, PAL["soft"]], [1, PRIMARY]], title="How busy each resource is")
    util.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="", coloraxis_showscale=False)
    c2.plotly_chart(util, width="stretch")
    show = rk.copy()
    show["Resource"] = [en(r) for r in show["Resource"]]
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Isolated delay = how long this resource alone would delay the project. Removal gain = days saved "
               "if it were unlimited. Impact = the larger of the two. Try the what-if crew sizes in the sidebar.")

# --------------------------------------------------------------------------- Gap 3
with tabs[2]:
    st.subheader("Which jobs fight over the same crews?")
    with st.spinner("Building the conflict graph..."):
        an, syn = conflicts(code, caps)
    m = st.columns(3)
    m[0].metric("Activities in conflict", len({a for e in an["edges"] for a in e}))
    m[1].metric("Conflicts (pairs)", len(an["edges"]))
    m[2].metric("Multi-resource activities", len(an["multi"]), "blocked by ≥2 resources at once", delta_color="off")

    if an["edges"]:
        G = nx.Graph()
        for (i, j), rs in an["edges"].items():
            G.add_edge(i, j, rs=rs)
        pos = nx.spring_layout(G, seed=7, k=0.9 / np.sqrt(max(G.number_of_nodes(), 1)))
        ex, ey = [], []
        for i, j in G.edges():
            ex += [pos[i][0], pos[j][0], None]; ey += [pos[i][1], pos[j][1], None]
        fig = go.Figure()
        fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=EDGE, width=1), hoverinfo="skip")
        nodes = list(G.nodes())
        multi = [n in an["multi"] for n in nodes]
        fig.add_scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
            text=nodes, textposition="top center", textfont_size=9,
            marker=dict(size=[8 + 1.2 * min(project.activities[n].duration, 20) for n in nodes],
                        color=[DELAY if mm else PRIMARY for mm in multi], line=dict(color=PAL["card"], width=1)),
            customdata=[[project.activities[n].name, project.activities[n].duration,
                         ", ".join(en(r) for r in sorted(set().union(*[G[n][x]["rs"] for x in G[n]])))] for n in nodes],
            hovertemplate="<b>%{text} · %{customdata[0]}</b><br>%{customdata[1]} days<br>competes for: %{customdata[2]}<extra></extra>")
        fig.update_layout(height=520, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is an activity (size = duration). Lines join activities that need the same crew on an "
                   "over-booked day. Terracotta = blocked by two or more resources at once.")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Conflict cores** (handed to the quantum solver, highest impact first)")
        st.dataframe(pd.DataFrame([{"Core": i, "Activities": len(c["activities"]),
                                    "Resources": ", ".join(en(r) for r in c["resources"]),
                                    "Impact (days)": c["impact"],
                                    "Multi-resource jobs": len(c["multi_resource_activities"])}
                                   for i, c in enumerate(an["cores"], 1)]), hide_index=True, width="stretch")
    with cr:
        st.markdown("**Multi-resource activities**")
        if an["multi"]:
            st.dataframe(pd.DataFrame([{"Activity": a, "Name": project.activities[a].name[:40],
                                        "Blocked by": " + ".join(en(r) for r in sorted(rs))}
                                       for a, rs in sorted(an["multi"].items())]), hide_index=True, width="stretch")
        else:
            st.info("This project's delay comes from one dominant crew; no multi-resource conflicts.")
        if len(syn):
            st.markdown("**Resource pairs** (days saved by relieving both)")
            s2 = syn.head(5).copy()
            s2["Resource pair"] = [" + ".join(en(x) for x in r.split(" + ")) for r in s2["Resource pair"]]
            st.dataframe(s2, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- Hybrid schedule
with tabs[3]:
    st.subheader("Hybrid quantum-classical schedule")
    st.markdown("Classical warm start → conflict-sized QUBO sub-problems solved by quantum annealing → "
                "every candidate rebuilt and checked against all rules.")
    run = st.button("▶ Run the hybrid scheduler", type="primary")
    key = (code, caps, budget)
    if run or st.session_state.get("hyb_key") == key:
        with st.spinner("Solving quantum sub-problems... (≈ 5-15 s)"):
            sched, hnotes, secs = schedule(code, caps, budget)
        st.session_state["hyb_key"] = key
        C, C0 = makespan(project, sched), int(hnotes["Warm-start makespan"])
        v = validate_schedule(project, sched)
        u = utilisation(project, sched)
        m = st.columns(5)
        m[0].metric("Project length", f"{C} days", f"{C - C0} days" if C != C0 else "same as initial",
                    delta_color="inverse")
        m[1].metric("Reduction", f"{100 * (C0 - C) / C0:.1f}%")
        m[2].metric("Quantum sub-problems", hnotes["Sub-QUBOs solved"])
        m[3].metric("Largest sub-problem", f"{hnotes['Max qubits/sub-QUBO']} qubits", f"budget {budget}", delta_color="off")
        m[4].metric("Rule violations", sum(v.values()))
        st.plotly_chart(gantt(project, sched, f"{project.info['label']}: {C}-day schedule"), width="stretch")
        res_order = [r for r in rk["Resource"]]
        r_sel = st.selectbox("Daily use of a crew / machine", res_order, format_func=en)
        st.plotly_chart(resource_chart(project, sched, r_sel), width="stretch")
        st.caption(f"Average utilisation {100 * sum(u.values()) / len(u):.1f}% · solved in {secs:.1f} s · "
                   "the blue area never crosses the red line: no crew is double-booked.")
        out = pd.DataFrame([{"id": a, "name": project.activities[a].name, "phase": project.activities[a].phase,
                             "start_day": s, "finish_day": s + project.activities[a].duration}
                            for a, s in sorted(sched.items(), key=lambda kv: (kv[1], kv[0]))])
        st.download_button("⬇ Download schedule (CSV)", out.to_csv(index=False), f"{code}_schedule.csv", "text/csv")
    else:
        st.info("Press **Run** to schedule this project. Change crew sizes or the qubit budget in the sidebar to explore what-ifs.")

# --------------------------------------------------------------------------- Quantum core
with tabs[4]:
    st.subheader("The quantum core: the worst conflict, solved whole")
    size = st.radio("Core size", ["laptop", "gpu"], horizontal=True,
                    format_func=lambda s: "16 qubits (laptop)" if s == "laptop" else "26 qubits (GPU)",
                    index=0 if aer_device() == "CPU" else 1)
    try:
        with st.spinner("Extracting the top conflict core..."):
            sp, q, info, opt = quantum_core(code, caps, size)
    except ValueError as e:
        st.warning(f"No conflict core fits this budget for this project ({e}).")
        st.stop()
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f"Extracted automatically from conflict core 1 · competing for "
                    f"**{', '.join(en(r) for r in info['resources'])}** "
                    f"(capacity left for this group: {', '.join(f'{en(r)} = {c}' for r, c in sp.capacity.items())}).")
        st.dataframe(pd.DataFrame([{"Activity": a.id, "Name": a.name[:36], "Days": a.duration,
                                    "Needs": ", ".join(f"{en(r)}×{u}" for r, u in a.demand.items())}
                                   for a in sp.activities.values()]), hide_index=True, width="stretch")
        m = st.columns(3)
        m[0].metric("Qubits", q.num_vars)
        m[1].metric("Slack qubits", q.num_slack)
        m[2].metric("Best possible", f"{opt} days")
    with c2:
        n = q.num_vars
        M = np.zeros((n, n))
        for v_, b in q.bqm.linear.items():
            M[v_, v_] = b
        for (u_, v_), b in q.bqm.quadratic.items():
            M[u_, v_] = M[v_, u_] = b
        hm = px.imshow(np.sign(M) * np.log1p(np.abs(M)), color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                       title="QUBO matrix (sign · log scale): diagonal = per-qubit cost, off-diagonal = penalties")
        hm.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(hm, width="stretch")

    b1, b2 = st.columns(2)
    if b1.button("⚡ Solve with quantum annealing (simulated)"):
        t = time.perf_counter()
        samp, _, ss = anneal(q, num_reads=300)
        sch, _ = q.decode(samp); sch.pop(END, None)
        st.session_state["sa"] = (code, caps, size, sch, time.perf_counter() - t)
    if b2.button("⚛️ Solve with QAOA (≈ 10-30 s)", type="primary"):
        with st.spinner(f"Tuning the QAOA circuit on the {aer_device()} simulator, then sampling 20,000 shots..."):
            out = run_qaoa(DATA, code, caps, size)  # separate process (see src/ui_jobs.py)
        st.session_state["qa"] = (code, caps, size, out["schedule"], out["stats"], out["random"], out["runtime_s"])

    sa = st.session_state.get("sa")
    if sa and sa[:3] == (code, caps, size):
        ok = is_feasible(sp, sa[3]) and len(sa[3]) == sp.n
        st.success(f"Quantum annealing: {makespan(sp, sa[3])}-day schedule, "
                   f"{'optimal ✓' if ok and makespan(sp, sa[3]) == opt else 'valid' if ok else 'invalid'} · {sa[4]:.2f} s")
    qa_ = st.session_state.get("qa")
    if qa_ and qa_[:3] == (code, caps, size):
        _, _, _, qsched, stats, rand, secs = qa_
        if qsched:
            st.success(f"QAOA: {makespan(sp, qsched)}-day schedule "
                       f"({'optimal ✓' if makespan(sp, qsched) == opt else 'valid'}) · {secs:.1f} s")
        g1, g2 = st.columns(2)
        with g1:
            rows = [("QAOA", 100 * stats["feasible_share"], 100 * stats["optimal_share"])]
            if rand:
                rows.insert(0, ("Random guessing", 100 * rand["feasible_share"], 100 * rand["optimal_share"]))
            df = pd.DataFrame(rows, columns=["Sampler", "Valid schedules (%)", "Optimal schedules (%)"])
            bars = px.bar(df.melt("Sampler", var_name="Share", value_name="%"), x="Sampler", y="%", color="Share",
                          barmode="group", log_y=True, color_discrete_sequence=[ACCENT, PRIMARY],
                          title="How often each sampler hits a valid / optimal schedule (log scale)")
            bars.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(bars, width="stretch")
            if rand and rand["feasible_share"] > 0:
                st.metric("QAOA vs random guessing", f"{stats['feasible_share'] / rand['feasible_share']:.0f}× more valid schedules")
        with g2:
            if qsched:
                st.plotly_chart(gantt(sp, qsched, "Schedule found by QAOA", height_per_row=40), width="stretch")

# --------------------------------------------------------------------------- Method
with tabs[5]:
    st.subheader("Method")
    st.markdown("**Problem.** Resource-Constrained Project Scheduling (RCPSP): pick a start day for every activity so "
                "every precedence link holds, no crew or machine is over-booked on any day, and the project finishes as "
                "early as possible.")
    st.markdown("**QUBO formulation** (time-indexed, Pritsker et al. 1969; Pérez Armas et al. 2024). "
                "Binary variable $x_{j,s}=1$ if activity $j$ starts on day $s$:")
    st.latex(r"E=\sum_t (t-t_{\min})\,e_t + P\Big[\sum_j\big(\textstyle\sum_s x_{j,s}-1\big)^2"
             r"+\sum_{i\to j}\sum_{s'<s+\delta_{ij}}x_{i,s}x_{j,s'}"
             r"+\sum_{k,t}\big(\text{use}_{k,t}+z_{k,t}-R_k\big)^2\Big]")
    st.markdown("""
| Research gap | What the code does |
|---|---|
| **1 · Bottleneck ranking** | For every resource: isolated delay (only it limited) and removal gain (it unlimited) → impact in days, ranked |
| **2 · Adaptive decomposition** | Sub-problems sized to each conflict; waiting jobs get more freedom; every sub-QUBO is shrunk to fit the qubit budget |
| **3 · Multi-resource conflicts** | Conflict graph labelled with every contested resource; multi-resource jobs; conflict cores feed the quantum core |
| **Quantum solvers** | Simulated quantum annealing (dwave-samplers) and QAOA (Qiskit, CPU/GPU simulator, or IBM hardware via `run_ibm.py`) |
""")
    st.markdown("**Data.** DSLIB v3.4 (Batselier & Vanhoucke, 2015, *Int. J. Project Management* 33(3)), "
                "Ghent University. Education Service is used exactly as recorded; the other two use availability "
                "reduced by the standard resource-strength rule (Kolisch et al., 1995).")
    st.markdown('<p class="note">All numbers in this app are computed live by the same code as run_demo.py.</p>',
                unsafe_allow_html=True)
