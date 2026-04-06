"""
VEN-Spaceflight Behavioral Analysis Pipeline
NASA OSD-618: Combined space stressors behavioral deficits

Maps OSD-618 behavioral assays to Fast Lane Hypothesis predictions:
  - Three Chamber Social Test  -> VEN social speed pathway
  - Novel Object Recognition   -> VEN rapid novelty detection
  - Radial Arm Water Maze -> VEN-mediated spatial SAT
  - Open Field -> locomotor confound control
  - Balance Beam -> sensorimotor confound control

Usage:
    python analysis/ven_spaceflight_analysis.py
"""

import os, sys, json
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
RES = ROOT / "results"
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

def assign_condition(row):
    rad  = str(row.get("Ionizing Radiation", "")).lower()
    hind = str(row.get("Hindlimb Unloading",  "")).lower()
    hous = str(row.get("Housing Condition",    "")).lower()
    is_rad  = "mixed" in rad  or "gcr" in rad  or "50" in rad
    is_hind = "unload" in hind
    is_iso  = "isolation" in hous
    if is_rad and (is_hind or is_iso):
        return "combined"
    elif is_rad:
        return "radiation"
    elif is_hind or is_iso:
        return "isolation"
    else:
        return "control"

def load_csv(fname):
    path = DATA / fname
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] File not found: {path}\n"
            f"Please download the TRANSFORMED CSV files from NASA OSDR OSD-618\n"
            f"into the data/ folder.\n"
        )
    return pd.read_csv(path)

def add_condition(df):
    if "Ionizing Radiation" in df.columns:
        df["condition"] = df.apply(assign_condition, axis=1)
    else:
        df["condition"] = "control"
    return df

def analyse_social(results):
    print("\n── Three Chamber Social Test ──")
    df = load_csv("LSDS-48_Three_Chamber_Social_Test_618_SOCIALAPPROACH_TRANSFORMED.csv")
    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")
    time_cols   = [c for c in df.columns if "time" in c.lower()]
    social_cols = [c for c in time_cols if any(k in c.lower() for k in ["mouse","social","stranger","novel_mouse"])]
    object_cols = [c for c in time_cols if any(k in c.lower() for k in ["object","cup","empty"])]
    df = add_condition(df)
    conds = ["control", "radiation", "isolation", "combined"]
    if social_cols and object_cols:
        sc, oc = social_cols[0], object_cols[0]
        df["spi"] = df[sc] / (df[sc] + df[oc] + 1e-9)
        metric, ylabel = "spi", "Social Preference Index"
    elif social_cols:
        metric, ylabel = social_cols[0], "Time Near Social (s)"
    else:
        metric = df.select_dtypes("number").columns[0]
        ylabel = metric
    summary = {}
    for c in conds:
        vals = df.loc[df["condition"] == c, metric].dropna()
        if len(vals):
            summary[c] = {"mean": float(vals.mean()), "sem": float(vals.sem()), "n": len(vals), "raw": vals.tolist()}
    groups = [np.array(summary[c]["raw"]) for c in conds if c in summary]
    if len(groups) >= 2:
        F, p = stats.f_oneway(*groups)
        print(f" ANOVA: F={F:.3f}, p={p:.4f}")
        for c in ["radiation","isolation","combined"]:
            if "control" in summary and c in summary:
                t, p2 = stats.ttest_ind(summary["control"]["raw"], summary[c]["raw"])
                print(f"   Control vs {c}: t={t:.3f}, p={p2:.4f}")
    results["social"] = {"metric": metric, "ylabel": ylabel, "summary": summary}
    return df, metric, ylabel, summary

def analyse_nor(results):
    print("\n- Novel Object Recognition -")
    df = load_csv("LSDS-48_Novel_Object_Recognition_618_NOR_TRANSFORMED.csv")
    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")
    novel_cols    = [c for c in df.columns if "novel" in c.lower() and any(k in c.lower() for k in ["time","explor"])]
    familiar_cols = [c for c in df.columns if any(k in c.lower() for k in ["familiar","object1","old"]) and any(k in c.lower() for k in ["time","explor"])]
    df = add_condition(df)
    conds = ["control", "radiation", "isolation", "combined"]
    if novel_cols and familiar_cols:
        nc, fc = novel_cols[0], familiar_cols[0]
        df["di"] = (df[nc] - df[fc]) / (df[nc] + df[fc] + 1e-9)
        metric, ylabel = "di", "Discrimination Index (DI)"
    else:
        num_cols = df.select_dtypes("number").columns.tolist()
        metric = num_cols[0] if num_cols else None
        ylabel = metric or "NOR metric"
    summary = {}
    if metric:
        for c in conds:
            vals = df.loc[df["condition"] == c, metric].dropna()
            if len(vals):
                summary[c] = {"mean": float(vals.mean()), "sem": float(vals.sem()), "n": len(vals), "raw": vals.tolist()}
    results["nor"] = {"metric": metric, "ylabel": ylabel, "summary": summary}
    return df, metric, ylabel, summary

def analyse_rawm(results):
    print("\n- Radial Arm Water Maze -")
    df = load_csv("LSDS-48_Radial_Arm_Water_Maze_618_RAWM_TRANSFORMED.csv")
    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")
    error_cols = [c for c in df.columns if "error" in c.lower()]
    lat_cols   = [c for c in df.columns if "latency" in c.lower() or "time" in c.lower()]
    df = add_condition(df)
    conds = ["control", "radiation", "isolation", "combined"]
    metric = error_cols[-1] if error_cols else (lat_cols[0] if lat_cols else None)
    ylabel = "Errors (final block)" if error_cols else "Latency (s)"
    summary = {}
    if metric:
        for c in conds:
            vals = df.loc[df["condition"] == c, metric].dropna()
            if len(vals):
                summary[c] = {"mean": float(vals.mean()), "sem": float(vals.sem()), "n": len(vals), "raw": vals.tolist()}
    results["rawm"] = {"metric": metric, "ylabel": ylabel, "summary": summary}
    return df, metric, ylabel, summary

def analyse_of(results):
    print("\n- Open Field -")
    df = load_csv("LSDS-48_Open_Field_618_OF_TRANSFORMED.csv")
    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")
    dist_cols = [c for c in df.columns if any(k in c.lower() for k in ["distance","dist","total_dist","locomot"])]
    df = add_condition(df)
    metric = dist_cols[0] if dist_cols else df.select_dtypes("number").columns[0]
    ylabel = "Total Distance (cm)"
    conds = ["control", "radiation", "isolation", "combined"]
    summary = {}
    for c in conds:
        vals = df.loc[df["condition"] == c, metric].dropna()
        if len(vals):
            summary[c] = {"mean": float(vals.mean()), "sem": float(vals.sem()), "n": len(vals)}
    results["open_field"] = {"metric": metric, "ylabel": ylabel, "summary": summary}
    return df, metric, ylabel, summary

def analyse_balance(results):
    print("\n- Balance Beam -")
    df = load_csv("LSDS-48_Balance_Beam_618_BalanceBeam_TRANSFORMED.csv")
    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ["time","latency","cross","traverse"])]
    df = add_condition(df)
    metric = time_cols[0] if time_cols else df.select_dtypes("number").columns[0]
    ylabel = "Traversal Time (s)"
    conds = ["control", "radiation", "isolation", "combined"]
    summary = {}
    for c in conds:
        vals = df.loc[df["condition"] == c, metric].dropna()
        if len(vals):
            summary[c] = {"mean": float(vals.mean()), "sem": float(vals.sem()), "n": len(vals)}
    results["balance"] = {"metric": metric, "ylabel": ylabel, "summary": summary}
    return df, metric, ylabel, summary

def plot_bar(ax, summary, ylabel, title, conds=None):
    if conds is None:
        conds = ["control", "radiation", "isolation", "combined"]
    x = np.arange(len(conds))
    means  = [summary.get(c, {}).get("mean", np.nan) for c in conds]
    sems   = [summary.get(c, {}).get("sem",  0.0)    for c in conds]
    colors = [COLORS.get(c, "#999") for c in conds]
    ax.bar(x, means, yerr=sems, color=colors, width=0.6, capsize=4,
           error_kw={"linewidth":1.5}, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

def make_overview_figure(res):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "VEN Fast Lane Hypothesis: Spaceflight Stressor Behavioral Signatures\n"
        "NASA OSD-618 — Combined Space Stressors (n=590 mice)",
        fontweight="bold", fontsize=13)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45)
    panels = [
        (gs[0,0], "social", "Social Approach", "Social Metric"),
        (gs[0,1], "nor", "Novel Object Recog.", "Discrimination Index"),
        (gs[0,2], "rawm", "Radial Arm Water Maze", "Errors / Latency"),
        (gs[1,0], "open_field", "Open Field (locomotion)", "Distance (cm)"),
        (gs[1,1], "balance", "Balance Beam (motor)", "Traversal Time (s)"),
    ]
    for pos, key, title, default_ylabel in panels:
        ax = fig.add_subplot(pos)
        entry   = res.get(key, {})
        summary = entry.get("summary", {})
        ylabel  = entry.get("ylabel", default_ylabel)
        if summary:
            plot_bar(ax, summary, ylabel, title)
        else:
            ax.text(0.5, 0.5, "data not found\ncheck data/ folder",
                    ha="center", va="center", transform=ax.transAxes, color="red", fontsize=9)
            ax.set_title(title, fontsize=11)
    ax_leg = fig.add_subplot(gs[1,2])
    ax_leg.axis("off")
    patches = [plt.Rectangle((0,0),1,1, color=COLORS[c])
               for c in ["control","radiation","isolation","combined"]]
    labels  = [COND_LABELS[c] for c in ["control","radiation","isolation","combined"]]
    ax_leg.legend(patches, labels, loc="center", fontsize=10,
                  title="Spaceflight Condition", title_fontsize=11)
    note = (
        "VEN Fast Lane Hypothesis predictions:\n"
        "• Social approach deficit -> VEN pathway stress\n"
        "• NOR discrimination loss -> rapid novelty signal impaired\n"
        "• RAWM errors ↑ -> spatial SAT disrupted\n"
        "• Open field / Balance Beam: motor confound controls\n\n"
        "Combined stressor (red) mirrors FTD-like\n"
        "VEN ablation in Fast Lane model."
    )
    ax_leg.text(0.5, 0.15, note, transform=ax_leg.transAxes,
                fontsize=8.5, va="bottom", ha="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f0f4ff", ec="#aabbdd"))
    out = FIGS / "fig_ven_spaceflight_overview.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

def make_deficit_heatmap(res):
    assays = ["social", "nor", "rawm", "open_field", "balance"]
    labels = ["Social Approach", "Novel Object Recog.", "Radial Arm Maze", "Open Field", "Balance Beam"]
    conds  = ["radiation", "isolation", "combined"]
    matrix = np.full((len(assays), len(conds)), np.nan)
    for i, assay in enumerate(assays):
        entry = res.get(assay, {})
        ctrl = entry.get("summary", {}).get("control", {})
        if not ctrl:
            continue
        ctrl_mean = ctrl["mean"]
        ctrl_sem  = ctrl.get("sem", 1.0) or 1.0
        for j, cond in enumerate(conds):
            cv = entry.get("summary", {}).get(cond, {}).get("mean")
            if cv is not None:
                matrix[i, j] = (cv - ctrl_mean) / (ctrl_sem + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=10)
    ax.set_yticks(range(len(assays)))
    ax.set_yticklabels(labels, fontsize=10)
    plt.colorbar(im, ax=ax, label="Z-score vs Control")
    ax.set_title("Behavioral Deficit Heatmap\n(Z-score relative to control)",
                 fontweight="bold", fontsize=12)
    for i in range(len(assays)):
        for j in range(len(conds)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color="black" if abs(v) < 2 else "white")
    plt.tight_layout()
    out = FIGS / "fig_deficit_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

def main():
    print("*" * 65)
    print("VEN Fast Lane Hypothesis - Spaceflight Behavioral Analysis")
    print("NASA OSD-618")
    print("*" * 65)
    results = {}
    for fn in [analyse_social, analyse_nor, analyse_rawm, analyse_of, analyse_balance]:
        try:
            fn(results)
        except FileNotFoundError as e:
            print(e)
    if results:
        make_overview_figure(results)
        make_deficit_heatmap(results)
    out_json = RES / "ven_spaceflight_results.json"
    serialisable = {}
    for k, v in results.items():
        entry = {"metric": v.get("metric"), "ylabel": v.get("ylabel"), "summary": {}}
        for c, s in v.get("summary", {}).items():
            entry["summary"][c] = {"mean": s.get("mean"), "sem": s.get("sem"), "n": s.get("n")}
        serialisable[k] = entry
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved: {out_json}")
    print("\nDone.")

if __name__ == "__main__":
    main()