"""
visualize.py -- Layer 5: charts (matplotlib only, saved as PNG, no pop-ups).

gantt()             one bar per activity on a day axis, coloured by phase
resource_profile()  daily usage of each resource vs its capacity
comparison()        makespan / utilisation / runtime for every method
qaoa_bars()         share of valid and optimal samples: QAOA vs random guessing
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .evaluate import makespan, usage_profile
from .project import Project

# English labels for the Dutch resource names in DSLIB C2012-02 (display only)
RESOURCE_EN = {
    "Herassen": "fence panels", "Vergaderkeet": "site cabin", "Graafkraan + bestuurder": "excavator",
    "Vrachtwagen": "truck", "Paalboorinstallatie + bestuurder": "pile-drilling rig", "Arbeider": "labourers",
    "Montagekraan + bestuurder": "assembly crane", "Betonpomp + bediener": "concrete pump",
    "Vlindertoestel": "power trowel", "Metser": "mason", "Hoogtewerker": "aerial platform",
    "Verreiker + bestuurder": "telehandler", "Dakwerkers": "roofers", "Veiligheidsinstallateur": "safety installer",
    "Staalmonteur": "steel erectors", "Elektricien": "electricians", "Stukadoor": "plasterer",
    "Schilder": "painter", "Loodgieter": "plumbers", "Techniekers Nut Mixing Station": "plant technicians",
    "Technieker Automatisering": "automation tech", "Brandtechnicus": "fire technicians",
    "Pneumatische drilhamer": "jackhammer", "Chauffagist": "heating installers",
}


def en(resource: str) -> str:
    return RESOURCE_EN.get(resource, resource)


def phase_colours(project: Project) -> Dict[str, str]:
    phases = []
    for aid in project.order:
        ph = project.activities[aid].phase or "Project"
        if ph not in phases:
            phases.append(ph)
    cmap = plt.get_cmap("tab20")
    return {ph: matplotlib.colors.to_hex(cmap(i % 20)) for i, ph in enumerate(phases)}


def gantt(project: Project, schedule: Dict[str, int], title: str, path: str,
          lower_bound: Optional[int] = None) -> None:
    ids = sorted(schedule, key=lambda a: (schedule[a], a))
    h = max(4, 0.22 * len(ids) + 1.5)
    fig, ax = plt.subplots(figsize=(12, h))
    colours = phase_colours(project)
    for row, aid in enumerate(ids):
        a = project.activities[aid]
        ax.barh(row, max(a.duration, 0.15), left=schedule[aid], height=0.7,
                color=colours[a.phase or "Project"], edgecolor="white")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{aid}  {project.activities[aid].name[:40]}" for aid in ids], fontsize=7)
    ax.invert_yaxis()
    C = makespan(project, schedule)
    ax.axvline(C, color="crimson", ls="--", lw=1.2, label=f"Makespan = {C} days")
    if lower_bound:
        ax.axvline(lower_bound, color="black", ls=":", lw=1.2, label=f"Proven optimum = {lower_bound}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    leg1 = ax.legend(handles, [k[:28] for k in colours], loc="lower left",
                     fontsize=7, title="Phase")
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Day")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def resource_profile(project: Project, schedule: Dict[str, int], title: str, path: str,
                     max_panels: int = 6) -> None:
    """Daily use vs capacity for the busiest resources."""
    prof = usage_profile(project, schedule)
    # busiest = most days at full capacity (ties: highest peak / capacity)
    rs = sorted(project.resources, key=lambda r: (-int((prof[r] >= project.capacity[r]).sum()),
                                                  -prof[r].max() / project.capacity[r]))[:max_panels]
    fig, axes = plt.subplots(len(rs), 1, figsize=(11, 1.7 * len(rs)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, r in zip(axes, rs):
        ax.step(range(len(prof[r])), prof[r], where="post", lw=1.4)
        ax.axhline(project.capacity[r], color="crimson", ls="--", lw=1, label="capacity")
        ax.set_ylabel(en(r)[:18], fontsize=8)
        ax.set_ylim(0, project.capacity[r] + 1)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Day")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def comparison(df: pd.DataFrame, lower_bound: int, path: str) -> None:
    labels = [m.split(" (")[0] for m in df["Method"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = range(len(labels))

    axes[0].bar(x, df["Makespan (days)"], color="#1f77b4")
    axes[0].axhline(lower_bound, color="black", ls=":", label=f"proven optimum {lower_bound}")
    for i, v in enumerate(df["Makespan (days)"]):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, df["Makespan (days)"].max() * 1.15)
    axes[0].set_title("Makespan (lower is better)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, df["Avg utilisation (%)"], color="#2ca02c")
    for i, v in enumerate(df["Avg utilisation (%)"]):
        axes[1].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Average resource utilisation % (higher is better)")

    axes[2].bar(x, df["Runtime (s)"], color="#ff7f0e")
    axes[2].set_yscale("log")
    axes[2].set_title("Runtime, seconds (log scale)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def qaoa_bars(rows: Dict[str, Dict[str, float]], path: str, qubits: int = 0) -> None:
    """rows: label -> {"feasible": %, "optimal": %}"""
    labels = list(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.38
    for k, (key, col) in enumerate([("feasible", "#2ca02c"), ("optimal", "#1f77b4")]):
        vals = [rows[l][key] for l in labels]
        xs = [i + (k - 0.5) * w for i in range(len(labels))]
        ax.bar(xs, vals, width=w, color=col, label=f"{key} samples (%)")
        for xx, v in zip(xs, vals):
            ax.text(xx, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("% of samples (log scale)")
    ax.set_title(f"Quantum slice ({qubits} qubits): share of samples that are valid / optimal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def bottleneck_chart(ranking: pd.DataFrame, title: str, path: str, top: int = 8) -> None:
    """GAP 1: how many days each resource delays the project, and how busy it is."""
    df = ranking.head(top).iloc[::-1]
    labels = [en(r)[:24] for r in df["Resource"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.45 * len(df) + 1.8), sharey=True)
    ax1.barh(labels, df["Impact (days)"], color="#d62728")
    for y, v in enumerate(df["Impact (days)"]):
        ax1.text(v, y, f" {v}", va="center", fontsize=8)
    ax1.set_xlabel("Delay caused (days)")
    ax1.set_title("Bottleneck impact")
    ax2.barh(labels, df["Utilisation (%)"], color="#1f77b4")
    for y, v in enumerate(df["Utilisation (%)"]):
        ax2.text(v, y, f" {v}", va="center", fontsize=8)
    ax2.set_xlabel("Utilisation (%)")
    ax2.set_title("How busy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def decomposition_chart(rows: pd.DataFrame, path: str) -> None:
    """GAP 2: fixed vs adaptive decomposition, per project."""
    projects = list(rows["Project"].unique())
    x = range(len(projects))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title in zip(axes, ["Makespan (days)", "Mean qubits/sub-QUBO", "Max qubits/sub-QUBO"],
                              ["Project length (days)", "Average sub-problem size (qubits)",
                               "Largest sub-problem (qubits)"]):
        for k, (dec, colour) in enumerate([("fixed window", "#9e9e9e"), ("adaptive", "#2ca02c")]):
            vals = [float(rows[(rows.Project == p) & (rows.Decomposition == dec)][col].iloc[0]) for p in projects]
            xs = [i + (k - 0.5) * 0.38 for i in x]
            ax.bar(xs, vals, width=0.38, color=colour, label=dec)
            for xx, v in zip(xs, vals):
                ax.text(xx, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([p.split(" ", 1)[1][:18] for p in projects], fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[2].axhline(156, color="crimson", ls="--", lw=1, label="free IBM machine (156 qubits)")
    axes[2].axhline(100, color="#2ca02c", ls=":", lw=1, label="adaptive budget (100 qubits)")
    axes[2].legend(fontsize=7, loc="upper left")
    axes[2].set_ylim(0, 230)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
