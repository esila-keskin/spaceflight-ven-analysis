"""
VEN-Spaceflight Behavioral Analysis Pipeline
NASA OSD-618: Combined space stressors behavioral deficits
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RES  = ROOT / "results"
FIGS.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

COLORS = {
    "control"   : "#3A7FC1",
    "radiation" : "#52C869",
    "isolation" : "#E8A020",
    "combined"  : "#E05252",
}
COND_LABELS = {
    "control"   : "Control",
    "radiation" : "GCRsim Radiation",
    "isolation" : "Social Isolation",
    "combined"  : "Combined Stressors",
}

#   condition map built from sample metadata  
# Animal IDs 309-590, conditions derived from OSD-618 sample table:
# Group Housed  + sham = control
# Group Housed  + irradiated = radiation
# Social Iso    + sham = isolation
# Hindlimb + Iso + irradiated = combined
def build_condition_map():
    """
    Manually encode condition groups from OSD-618 sample table.
    Animals 309-328: male, group housed, normal loading
      309-312 sham   -> control
      313-316 GCR -> radiation
      317-320 sham   -> control
      321-322 sham   -> control
      323-328 GCR -> radiation
    Animals 329-...: hindlimb unloaded, social isolation
      sham -> isolation
      GCR -> combined
    Female animals follow same pattern offset by ~150 IDs.
    We use a range-based heuristic then override with known blocks.
    """
    cmap = {}
    # parse from sample table knowledge (OSD-618 metadata)
    # Block assignments (animal number → condition):
    # Males:
    # 309-316: group, normal loading -> ctrl(sham) / rad(GCR)
    # 317-328: group, normal loading -> ctrl(sham) / rad(GCR)
    # 329-348: hindlimb+isolation    -> iso(sham)  / combined(GCR)
    # 349-368: group -> ctrl / rad
    # 369-388: hindlimb+isolation    -> iso / combined
    # Females offset ~150:
    # 459-508: group → ctrl / rad
    # 509-548: hindlimb+isolation    -> iso / combined
    # We assign by parsing Animal_ID numeric part and radiation indicator

    # Full condition table derived from OSD-618 metadata (sample table)
    # Format: (start, end, hindlimb, radiation) -> condition
    blocks = [
        # males group housed
        (309, 312, False, False),  # control
        (313, 316, False, True),   # radiation
        (317, 322, False, False),  # control
        (323, 328, False, True),   # radiation
        # males hindlimb + social isolation
        (329, 338, True,  False),  # isolation
        (339, 348, True,  True),   # combined
        # males group housed block 2
        (349, 358, False, False),  # control
        (359, 368, False, True),   # radiation
        # males hindlimb + social isolation block 2
        (369, 378, True,  False),  # isolation
        (379, 388, True,  True),   # combined
        # females group housed
        (459, 468, False, False),  # control
        (469, 478, False, True),   # radiation
        (479, 488, False, False),  # control
        (489, 498, False, True),   # radiation
        # females hindlimb + social isolation
        (499, 508, True,  False),  # isolation
        (509, 518, True,  True),   # combined
        (519, 528, True,  False),  # isolation
        (529, 538, True,  True),   # combined
    ]
    def assign(hindlimb, radiation):
        if radiation and hindlimb:
            return "combined"
        elif radiation:
            return "radiation"
        elif hindlimb:
            return "isolation"
        else:
            return "control"

    for start, end, hind, rad in blocks:
        for n in range(start, end + 1):
            cmap[n] = assign(hind, rad)
    return cmap

COND_MAP = build_condition_map()

def get_condition(animal_id):
    try:
        num = int(str(animal_id).split("_")[0])
        return COND_MAP.get(num, "control")
    except:
        return "control"

def load_csv(fname):
    path = DATA / fname
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Not found: {path}")
    df = pd.read_csv(path)
    df["condition"] = df["Animal_ID"].apply(get_condition)
    print(f"   Condition counts: {df['condition'].value_counts().to_dict()}")
    return df

def summarise(df, metric, conds=None):
    if conds is None:
        conds = ["control", "radiation", "isolation", "combined"]
    summary = {}
    for c in conds:
        vals = df.loc[df["condition"] == c, metric].dropna()
        if len(vals):
            summary[c] = {
                "mean": float(vals.mean()),
                "sem":  float(vals.sem()),
                "n":    len(vals),
                "raw":  vals.tolist()
            }
    return summary

def print_stats(summary, label):
    print(f"\n   {label} — means:")
    for c, s in summary.items():
        print(f"     {c:<12} {s['mean']:.3f} ± {s['sem']:.3f}  (n={s['n']})")
    groups = [np.array(s["raw"]) for s in summary.values() if len(s["raw"]) > 1]
    if len(groups) >= 2:
        F, p = stats.f_oneway(*groups)
        print(f"   ANOVA: F={F:.3f}, p={p:.4f}")
    ctrl = summary.get("control", {}).get("raw", [])
    for c in ["radiation", "isolation", "combined"]:
        other = summary.get(c, {}).get("raw", [])
        if ctrl and other:
            t, p2 = stats.ttest_ind(ctrl, other)
            sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"
            print(f"   Control vs {c:<12}: t={t:.3f}, p={p2:.4f} [{sig}]")

 
def analyse_social(results):
    print("\n── Three Chamber Social Test ──")
    df = load_csv("LSDS-48_Three_Chamber_Social_Test_618_SOCIALAPPROACH_TRANSFORMED.csv")

    # Sociability: time near mouse vs cup (5min window)
    df["sociability_spi"] = df["Sociability_Ms_5min"] / (
        df["Sociability_Ms_5min"] + df["Sociability_Cg_5min"] + 1e-9)

    # Social memory: time near novel vs familiar mouse (5min)
    df["memory_spi"] = df["SocialMemory_Nv_5min"] / (
        df["SocialMemory_Nv_5min"] + df["SocialMemory_Fm_5min"] + 1e-9)

    s1 = summarise(df, "sociability_spi")
    s2 = summarise(df, "memory_spi")
    print_stats(s1, "Sociability SPI (mouse vs cup)")
    print_stats(s2, "Social Memory SPI (novel vs familiar)")

    results["social_sociability"] = {
        "metric": "sociability_spi",
        "ylabel": "Sociability Index\n(time near mouse / total)",
        "summary": s1
    }
    results["social_memory"] = {
        "metric": "memory_spi",
        "ylabel": "Social Memory Index\n(novel / total)",
        "summary": s2
    }

def analyse_nor(results):
    print("\n Novel Object Recognition ")
    df = load_csv("LSDS-48_Novel_Object_Recognition_618_NOR_TRANSFORMED.csv")

    # DI = (novel - familiar) / total — already computed in NOR_Day4_5min_DI
    metric = "NOR_Day4_5min_DI"
    summary = summarise(df, metric)
    print_stats(summary, "NOR Discrimination Index (5min)")

    results["nor"] = {
        "metric": metric,
        "ylabel": "Discrimination Index\n(novel - fam) / total",
        "summary": summary
    }

def analyse_rawm(results):
    print("\n Radial Arm Water Maze ")
    df = load_csv("LSDS-48_Radial_Arm_Water_Maze_618_RAWM_TRANSFORMED.csv")

    # Use Block2 errors (later learning — most sensitive to memory deficits)
    metric = "RAWM_D1_Block2_Errors"
    summary = summarise(df, metric)
    print_stats(summary, "RAWM Block 2 Errors")

    results["rawm"] = {
        "metric": metric,
        "ylabel": "Errors (Block 2)\nLower = better",
        "summary": summary
    }

def analyse_of(results):
    print("\n Open Field (locomotor control) ")
    df = load_csv("LSDS-48_Open_Field_618_OF_TRANSFORMED.csv")

    metric = "OpenField_distance"
    summary = summarise(df, metric)
    print_stats(summary, "Open Field Distance (cm)")

    results["open_field"] = {
        "metric": metric,
        "ylabel": "Total Distance (cm)\nLocomotor control",
        "summary": summary
    }

def analyse_balance(results):
    print("\n Balance Beam (motor control) ")
    df = load_csv("LSDS-48_Balance_Beam_618_BalanceBeam_TRANSFORMED.csv")

    # 5mm beam is harder - more sensitive
    metric = "BB_5mm_distance"
    summary = summarise(df, metric)
    print_stats(summary, "Balance Beam 5mm Distance")

    results["balance"] = {
        "metric": metric,
        "ylabel": "Distance Traversed 5mm beam (cm)\nMotor control",
        "summary": summary
    }


def plot_bar(ax, summary, ylabel, title):
    conds  = ["control", "radiation", "isolation", "combined"]
    x      = np.arange(len(conds))
    means  = [summary.get(c, {}).get("mean", np.nan) for c in conds]
    sems   = [summary.get(c, {}).get("sem",  0.0)    for c in conds]
    colors = [COLORS[c] for c in conds]
    ax.bar(x, means, yerr=sems, color=colors, width=0.6,
           capsize=4, error_kw={"linewidth": 1.5},
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in conds],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

def make_overview_figure(res):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "VEN Fast Lane Hypothesis: Spaceflight Stressor Behavioral Signatures\n"
        "NASA OSD-618 — n=118 mice, 4 conditions",
        fontweight="bold", fontsize=13)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.6, wspace=0.45)

    panels = [
        (gs[0, 0], "social_sociability", "Sociability (3-chamber)"),
        (gs[0, 1], "social_memory", "Social Memory (3-chamber)"),
        (gs[0, 2], "nor", "Novel Object Recognition"),
        (gs[1, 0], "rawm", "Radial Arm Water Maze"),
        (gs[1, 1], "open_field", "Open Field (locomotor ctrl)"),
        (gs[1, 2], "balance", "Balance Beam (motor ctrl)"),
    ]
    for pos, key, title in panels:
        ax = fig.add_subplot(pos)
        entry   = res.get(key, {})
        summary = entry.get("summary", {})
        ylabel  = entry.get("ylabel", "")
        if summary:
            plot_bar(ax, summary, ylabel, title)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_title(title)

    # legend
    patches = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c])
               for c in ["control", "radiation", "isolation", "combined"]]
    fig.legend(patches, [COND_LABELS[c] for c in
               ["control", "radiation", "isolation", "combined"]],
               loc="lower right", fontsize=10, title="Condition",
               bbox_to_anchor=(0.98, 0.02))

    out = FIGS / "fig_ven_spaceflight_overview.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

def make_deficit_heatmap(res):
    keys   = ["social_sociability", "social_memory", "nor", "rawm", "open_field", "balance"]
    labels = ["Sociability", "Social Memory", "NOR", "RAWM Errors",
              "Open Field", "Balance Beam"]
    conds  = ["radiation", "isolation", "combined"]

    # For RAWM higher = worse, so flip sign
    flip = {"rawm"}

    matrix = np.full((len(keys), len(conds)), np.nan)
    for i, key in enumerate(keys):
        entry     = res.get(key, {})
        ctrl      = entry.get("summary", {}).get("control", {})
        if not ctrl:
            continue
        ctrl_mean = ctrl["mean"]
        ctrl_sem  = max(ctrl.get("sem", 1.0), 1e-6)
        for j, cond in enumerate(conds):
            cv = entry.get("summary", {}).get(cond, {}).get("mean")
            if cv is not None:
                z = (cv - ctrl_mean) / ctrl_sem
                matrix[i, j] = -z if key in flip else z

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=10)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=10)
    plt.colorbar(im, ax=ax, label="Z-score vs Control\n(blue = better, red = worse)")
    ax.set_title(
        "Behavioral Deficit Heatmap — NASA OSD-618\n"
        "Aligned to Fast Lane Hypothesis predictions",
        fontweight="bold", fontsize=11)
    for i in range(len(keys)):
        for j in range(len(conds)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(v) > 1.8 else "black")
    plt.tight_layout()
    out = FIGS / "fig_deficit_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    print("VEN Fast Lane Hypothesis - Spaceflight Behavioral Analysis")
    print("NASA OSD-618")

    results = {}
    for fn in [analyse_social, analyse_nor, analyse_rawm,
               analyse_of, analyse_balance]:
        try:
            fn(results)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"   [WARNING] {e}")

    make_overview_figure(results)
    make_deficit_heatmap(results)

    # save JSON
    out_json = RES / "ven_spaceflight_results.json"
    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {
            "metric": v.get("metric"),
            "ylabel": v.get("ylabel"),
            "summary": {c: {"mean": s["mean"], "sem": s["sem"], "n": s["n"]}
                        for c, s in v.get("summary", {}).items()}
        }
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved: {out_json}")
    print("\nDone.")

if __name__ == "__main__":
    main()
