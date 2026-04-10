"""
VEN-Spaceflight Behavioral Analysis Pipeline
NASA OSD-618: Combined space stressors behavioral deficits

Data source: https://osdr.nasa.gov/bio/repo/data/studies/OSD-618

Required files (place in data/ directory):
  1. OSD-618_metadata_OSD-618-ISA.zip
  2. LSDS-48_Three_Chamber_Social_Test_618_SOCIALAPPROACH_TRANSFORMED.csv
  3. LSDS-48_Novel_Object_Recognition_618_NOR_TRANSFORMED.csv
  4. LSDS-48_Radial_Arm_Water_Maze_618_RAWM_TRANSFORMED.csv
  5. LSDS-48_Open_Field_618_OF_TRANSFORMED.csv
  6. LSDS-48_Balance_Beam_618_BalanceBeam_TRANSFORMED.csv 

Design (from Rienecker et al. 2023, doi:10.1038/s41598-023-28508-0):
  Factors: Housing (Group / SI / SI+HU)  x  Radiation (sham / GCRsim 50cGy)  x  Sex
  n = 118 animals (subset of 590 samples with behavioral assay data)
  Female mice were largely resilient to deficits observed in males.

Condition labels used here:
  GH_sham  = Group Housed, sham-irradiated (reference control)
  GH_GCR = Group Housed,  GCRsim 50 cGy
  SI_sham  = Social Isolation (normal loading),  sham
  SI_GCR = Social Isolation (normal loading),  GCRsim
  HU_sham  = Social Isolation + Hindlimb Unloaded, sham
  HU_GCR = Social Isolation + Hindlimb Unloaded, GCRsim
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RES  = ROOT / "results"
FIGS.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "GH_sham": "#3A7FC1",
    "GH_GCR":  "#52C869",
    "SI_sham": "#E8A020",
    "SI_GCR":  "#E05252",
    "HU_sham": "#9B59B6",
    "HU_GCR":  "#C0392B",
}
COND_LABELS = {
    "GH_sham": "GH Sham",
    "GH_GCR":  "GH GCRsim",
    "SI_sham": "SI Sham",
    "SI_GCR":  "SI GCRsim",
    "HU_sham": "SI+HU Sham",
    "HU_GCR":  "SI+HU GCRsim",
}
ORDERED_CONDITIONS = ["GH_sham", "GH_GCR", "SI_sham", "SI_GCR", "HU_sham", "HU_GCR"]
REFERENCE_CONDITION = "GH_sham"

BEHAVIORAL_FILES = {
    "social": "LSDS-48_Three_Chamber_Social_Test_618_SOCIALAPPROACH_TRANSFORMED.csv",
    "nor": "LSDS-48_Novel_Object_Recognition_618_NOR_TRANSFORMED.csv",
    "rawm": "LSDS-48_Radial_Arm_Water_Maze_618_RAWM_TRANSFORMED.csv",
    "of": "LSDS-48_Open_Field_618_OF_TRANSFORMED.csv",
    "balance": "LSDS-48_Balance_Beam_618_BalanceBeam_TRANSFORMED.csv",
}

#load and parse ISA metadata
def load_isa_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Open OSD-618_metadata_OSD-618-ISA.zip, find the sample table (s_*.txt),
    and return a tidy DataFrame with one row per animal.

    Columns returned:
        animal_num int e.g. 309
        sex str 'male' | 'female'
        hindlimb str raw ISA value
        radiation str raw ISA value
        housing str raw ISA value
        condition str derived label (like 'GH_sham')

    Raises
    FileNotFoundError  if the zip is missing - tells the user exactly what to download
    ValueError if the zip exists but expected columns are absent
    """
    zip_path = data_dir / "OSD-618_metadata_OSD-618-ISA.zip"

    if not zip_path.exists():
        raise FileNotFoundError(
            "[FATAL] Metadata zip not found:\n"
            f" Expected: {zip_path}\n\n"
        )

    # Locate the sample table inside the zip (ISA-Tab: starts with 's_')
    with zipfile.ZipFile(zip_path, "r") as zf:
        sample_files = [
            n for n in zf.namelist()
            if Path(n).name.startswith("s_") and n.endswith(".txt")
        ]
        if not sample_files:
            raise ValueError(
                f"No sample table (s_*.txt) found in {zip_path.name}.\n"
                f"Contents: {zf.namelist()}"
            )
        sample_file = sample_files[0]
        print(f"  Parsing sample table: {sample_file}")
        with zf.open(sample_file) as f:
            raw = f.read().decode("utf-8")

    # ISA-Tab is tab-separated
    meta = pd.read_csv(io.StringIO(raw), sep="\t", dtype=str)
    meta.columns = meta.columns.str.strip()

    print(f"  Metadata rows: {len(meta)}, columns: {len(meta.columns)}")

    def find_col(df, *substrings):
        """Return the first column containing any of the given substrings."""
        for sub in substrings:
            matches = [c for c in df.columns if sub.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    col_source = find_col(meta, "Source Name")
    col_sex = find_col(meta, "Factor Value[Sex]", "Factor Value: Sex")
    col_hindlimb = find_col(meta, "Factor Value[Hindlimb", "Factor Value: Hindlimb")
    col_radiation = find_col(meta, "Factor Value[Ionizing Radiation]",
                             "Factor Value: Ionizing Radiation")
    col_housing = find_col(meta, "Factor Value[Housing Condition]",
                             "Factor Value: Housing Condition")

    missing = [
        label for label, col in [
            ("Source Name", col_source),
            ("Sex", col_sex),
            ("Hindlimb Unloading",col_hindlimb),
            ("Ionizing Radiation",col_radiation),
            ("Housing Condition", col_housing),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not locate these columns in {sample_file}: {missing}\n"
            f"Available columns:\n  " + "\n  ".join(meta.columns.tolist())
        )

    # build tidy metadata table  
    tidy = pd.DataFrame({
        "source_name": meta[col_source].str.strip(),
        "sex": meta[col_sex].str.strip().str.lower(),
        "hindlimb": meta[col_hindlimb].str.strip(),
        "radiation": meta[col_radiation].str.strip(),
        "housing": meta[col_housing].str.strip(),
    })

    # Source Name is the raw animal number (like "309")
    tidy["animal_num"] = pd.to_numeric(tidy["source_name"], errors="coerce")
    tidy = tidy.dropna(subset=["animal_num"]).copy()
    tidy["animal_num"] = tidy["animal_num"].astype(int)

    # derive condition from the three independent factors 
    # Housing level:
    # Hindlimb Unloaded : HU  (all HU animals are also socially isolated)
    # Social Isolation + Normal Loading : SI
    # Group Housed + Normal Loading : GH  (reference)
    # Radiation level:
    # sham-irradiated : sham
    # mixed radiation field : GCR
    def assign_condition(row):
        h_raw = row["hindlimb"]
        r_raw = row["radiation"]
        c_raw = row["housing"]

        if "hindlimb unloaded" in h_raw.lower():
            housing_level = "HU"
        elif "social isolation" in c_raw.lower():
            housing_level = "SI"
        elif "group housed" in c_raw.lower():
            housing_level = "GH"
        else:
            return "UNKNOWN"  # will trigger a warning below

        rad_level = "sham" if "sham" in r_raw.lower() else "GCR"
        return f"{housing_level}_{rad_level}"

    tidy["condition"] = tidy.apply(assign_condition, axis=1)

    unknown = tidy[tidy["condition"] == "UNKNOWN"]
    if len(unknown) > 0:
        print(f"\n WARNING: {len(unknown)} rows have unrecognised factor values:")
        print(unknown[["animal_num", "hindlimb", "housing", "radiation"]].to_string())
        print(" These will be excluded from analysis.\n")
    tidy = tidy[tidy["condition"] != "UNKNOWN"].copy()

    # drop duplicates: ISA sample tables can list the same source animal
    # multiple times (once per assay type). Keep first occurrence.
    n_before = len(tidy)
    tidy = tidy.drop_duplicates(subset="animal_num", keep="first")
    n_after = len(tidy)
    if n_before != n_after:
        print(f" Dropped {n_before - n_after} duplicate animal entries "
              f"(expected - one ISA row per assay type).")

    print(f"\n  Unique animals with metadata: {len(tidy)}")
    print(" Condition × Sex distribution:")
    dist = tidy.groupby(["condition", "sex"]).size().rename("n")
    print(dist.to_string())
    print()

    return tidy[["animal_num", "sex", "hindlimb", "radiation", "housing", "condition"]]


 # load behavioral CSV and join with metadata

def load_behavioral(fname: str, meta: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Load a TRANSFORMED behavioral CSV, extract the numeric animal ID,
    and left-join with the metadata table.

    Raises
    FileNotFoundError  if the CSV is missing
    RuntimeError if zero animals match metadata (likely ID format mismatch)
    """
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Behavioral file not found: {path}\n"
            "Download from the OSDR OSD-618 study page (behavioral assay section)."
        )

    df = pd.read_csv(path)

    if "Animal_ID" not in df.columns:
        raise ValueError(
            f"'Animal_ID' column missing from {fname}.\n"
            f"Columns present: {df.columns.tolist()}"
        )

    # Animal_ID format is "309_39week" , extract the leading integer
    df["animal_num"] = df["Animal_ID"].apply(
        lambda x: int(str(x).split("_")[0])
    )

    before = len(df)
    merged = df.merge(meta, on="animal_num", how="left")

    # Hard fail if no animals matched at all - this would indicate a fundamental mismatch between files (wrong study, wrong ID format)
    n_matched = merged["condition"].notna().sum()
    if n_matched == 0:
        raise RuntimeError(
            f"[FATAL] Zero animals in {fname} matched the metadata.\n"
            "Possible causes:\n"
            "  - Behavioral file is from a different study\n"
            "  - Animal IDs in the CSV do not match Source Name in the ISA table\n"
            "Check both files manually before proceeding."
        )

    # Warn (but continue) if some animals are unmatched - could be legitimately excluded animals (e.g. excluded from a specific assay)
    unmatched = merged[merged["condition"].isna()]
    if len(unmatched) > 0:
        print(f"  NOTE: {len(unmatched)}/{before} animals in {fname} "
              f"not found in metadata and will be excluded: "
              f"{sorted(unmatched['animal_num'].tolist())}")
        merged = merged[merged["condition"].notna()].copy()

    print(f"  {fname.split('_')[0]}...csv: {len(merged)} animals matched")
    print("  " + merged.groupby(["sex", "condition"]).size()
          .rename("n").to_string().replace("\n", "\n  "))
    print()

    return merged


def summarise(df: pd.DataFrame, metric: str) -> dict:
    """
    Return summary[sex][condition] = {mean, sem, n, raw}
    for every (sex, condition) combination present in df.
    """
    summary: dict = {}
    for sex in sorted(df["sex"].dropna().unique()):
        summary[sex] = {}
        sub = df[df["sex"] == sex]
        for cond in [c for c in ORDERED_CONDITIONS
                     if c in sub["condition"].values]:
            vals = sub.loc[sub["condition"] == cond, metric].dropna()
            if len(vals) == 0:
                continue
            summary[sex][cond] = {
                "mean": float(vals.mean()),
                "sem":  float(vals.sem()) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
                "raw":  vals.tolist(),
            }
    return summary


def print_stats(summary: dict, label: str, higher_is_worse: bool = False):
    """
    Print per-sex means, one-way ANOVA, and Welch t-tests vs reference.
    higher_is_worse=True flips the direction label for error-count metrics.
    """
    print(f"\n {label}")
    for sex in sorted(summary.keys()):
        cond_data = summary[sex]
        if not cond_data:
            continue
        print(f"\n  Sex: {sex}  (reference = {REFERENCE_CONDITION})")
        for cond, s in cond_data.items():
            print(f" {cond:<12}  mean={s['mean']:.3f} ± {s['sem']:.3f}  (n={s['n']})")

        # One-way ANOVA across all conditions for this sex
        groups = [np.array(s["raw"]) for s in cond_data.values() if s["n"] > 1]
        if len(groups) >= 2:
            F, p = stats.f_oneway(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f" One-way ANOVA: F={F:.3f}, p={p:.4f} [{sig}]")

        # Welch t-tests vs reference condition
        ref_data = cond_data.get(REFERENCE_CONDITION, {})
        if not ref_data:
            print(f" (reference '{REFERENCE_CONDITION}' absent for {sex} - "
                  "skipping pairwise tests)")
            continue
        ref_vals = np.array(ref_data["raw"])

        for cond, s in cond_data.items():
            if cond == REFERENCE_CONDITION:
                continue
            if s["n"] < 2:
                continue
            other = np.array(s["raw"])
            t, p2 = stats.ttest_ind(ref_vals, other, equal_var=False)
            sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "ns"
            if higher_is_worse:
                direction = "↑ worse" if np.mean(other) > np.mean(ref_vals) else "↓ better"
            else:
                direction = "↑ better" if np.mean(other) > np.mean(ref_vals) else "↓ worse"
            print(f" {REFERENCE_CONDITION} vs {cond:<12}: "
                  f"t={t:+.3f}, p={p2:.4f} [{sig}] {direction}")


def find_col(df: pd.DataFrame, *substrings) -> str | None:
    """Return first column name containing any of the given substrings."""
    for sub in substrings:
        matches = [c for c in df.columns if sub.lower() in c.lower()]
        if matches:
            return matches[0]
    return None


 #per-assay analyses

def analyse_social(meta: pd.DataFrame, results: dict):
    print("\n Three Chamber Social Approach")
    df = load_behavioral(BEHAVIORAL_FILES["social"], meta, DATA)

    col_ms = find_col(df, "Sociability_Ms_5min")
    col_cg = find_col(df, "Sociability_Cg_5min")
    col_nv = find_col(df, "SocialMemory_Nv_5min")
    col_fm = find_col(df, "SocialMemory_Fm_5min")

    missing = [n for n, c in [("Ms_5min", col_ms), ("Cg_5min", col_cg),
                               ("Nv_5min", col_nv), ("Fm_5min", col_fm)]
               if c is None]
    if missing:
        raise ValueError(f"Missing social approach columns: {missing}. "
                         f"Available: {df.columns.tolist()}")

    # Social Preference Index: fraction of time near mouse vs total
    df["sociability_spi"] = df[col_ms] / (df[col_ms] + df[col_cg] + 1e-9)
    # Social Memory Index: fraction of time near novel vs total
    df["memory_spi"] = df[col_nv] / (df[col_nv] + df[col_fm] + 1e-9)

    s1 = summarise(df, "sociability_spi")
    s2 = summarise(df, "memory_spi")
    print_stats(s1, "Sociability SPI (mouse / total)")
    print_stats(s2, "Social Memory SPI (novel / total)")

    results["social_sociability"] = {
        "metric": "sociability_spi",
        "ylabel": "Sociability Index\n(mouse / total time)",
        "summary": s1,
        "higher_is_worse": False,
    }
    results["social_memory"] = {
        "metric": "memory_spi",
        "ylabel": "Social Memory Index\n(novel / total time)",
        "summary": s2,
        "higher_is_worse": False,
    }


def analyse_nor(meta: pd.DataFrame, results: dict):
    print("\n Novel Object Recognition ")
    df = load_behavioral(BEHAVIORAL_FILES["nor"], meta, DATA)

    # Find Discrimination Index column (DI = (novel - familiar) / total)
    metric = find_col(df, "NOR_Day4_5min_DI", "5min_DI", "_DI")
    if metric is None:
        di_candidates = [c for c in df.columns if "DI" in c.upper()]
        if not di_candidates:
            raise ValueError(f"No DI column found in NOR data. "
                             f"Columns: {df.columns.tolist()}")
        metric = di_candidates[0]
    print(f"  Using NOR metric: {metric}")

    summary = summarise(df, metric)
    print_stats(summary, f"NOR Discrimination Index")
    results["nor"] = {
        "metric": metric,
        "ylabel": "Discrimination Index\n(novel − fam) / total",
        "summary": summary,
        "higher_is_worse": False,
    }


def analyse_rawm(meta: pd.DataFrame, results: dict):
    print("\n Radial Arm Water Maze")
    df = load_behavioral(BEHAVIORAL_FILES["rawm"], meta, DATA)

    error_cols = [c for c in df.columns if "error" in c.lower()]
    if not error_cols:
        raise ValueError(f"No error columns found in RAWM data. "
                         f"Columns: {df.columns.tolist()}")
    print(f"  Available error columns: {error_cols}")

    # Prefer Day 1 Block 2 (acquisition errors, most sensitive to memory deficit)
    preferred = [c for c in error_cols
                 if ("D1" in c or "Day1" in c) and "Block2" in c]
    metric = preferred[0] if preferred else error_cols[0]
    print(f"  Using RAWM metric: {metric}")

    summary = summarise(df, metric)
    print_stats(summary, f"RAWM Errors ({metric})", higher_is_worse=True)
    results["rawm"] = {
        "metric": metric,
        "ylabel": f"Entry Errors\n(lower = better)",
        "summary": summary,
        "higher_is_worse": True,
    }


def analyse_of(meta: pd.DataFrame, results: dict):
    print("\n Open Field (locomotor control)")
    df = load_behavioral(BEHAVIORAL_FILES["of"], meta, DATA)

    metric = find_col(df, "OpenField_distance", "distance", "Distance")
    if metric is None:
        raise ValueError(f"No distance column found in Open Field data. "
                         f"Columns: {df.columns.tolist()}")
    print(f"  Using Open Field metric: {metric}")

    summary = summarise(df, metric)
    print_stats(summary, "Open Field Total Distance (cm)")
    results["open_field"] = {
        "metric": metric,
        "ylabel": "Total Distance (cm)\n(locomotor control)",
        "summary": summary,
        "higher_is_worse": False,
    }


def analyse_balance(meta: pd.DataFrame, results: dict):
    print("\n Balance Beam (motor control)")
    df = load_behavioral(BEHAVIORAL_FILES["balance"], meta, DATA)

    dist_cols = [c for c in df.columns
                 if "distance" in c.lower() or "Distance" in c]
    if not dist_cols:
        raise ValueError(f"No distance columns found in Balance Beam data. "
                         f"Columns: {df.columns.tolist()}")
    print(f"  Available distance columns: {dist_cols}")

    # 5 mm beam: harder, more sensitive to motor deficits
    mm5 = [c for c in dist_cols if "5mm" in c.lower() or "5_mm" in c.lower()]
    metric = mm5[0] if mm5 else dist_cols[0]
    print(f"  Using Balance Beam metric: {metric}")

    summary = summarise(df, metric)
    print_stats(summary, f"Balance Beam Distance ({metric})")
    results["balance"] = {
        "metric": metric,
        "ylabel": "Distance Traversed (cm)\n5 mm beam",
        "summary": summary,
        "higher_is_worse": False,
    }
 
def plot_bar_bysex(ax, summary: dict, ylabel: str, title: str,
                   higher_is_worse: bool = False):
    """
    Grouped bar chart: conditions on x-axis, male/female side by side.
    Solid bars = male; hatched bars = female.
    """
    sexes = sorted(summary.keys())
    # Only include conditions present in at least one sex
    all_conds = [c for c in ORDERED_CONDITIONS
                 if any(c in summary.get(s, {}) for s in sexes)]
    if not all_conds:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="red")
        ax.set_title(title, fontsize=9, fontweight="bold")
        return

    n_cond  = len(all_conds)
    n_sex = len(sexes)
    width = 0.35
    x = np.arange(n_cond)
    offsets = np.linspace(-(n_sex - 1) * width / 2,
                           (n_sex - 1) * width / 2, n_sex)
    hatches = ["", "///"]
    alphas  = [0.90, 0.60]

    for i, sex in enumerate(sexes):
        means, sems, colors = [], [], []
        for cond in all_conds:
            d = summary[sex].get(cond, {})
            means.append(d.get("mean", np.nan))
            sems.append(d.get("sem", 0.0))
            colors.append(COLORS.get(cond, "#888888"))

        ax.bar(
            x + offsets[i], means, width,
            yerr=sems,
            color=colors,
            alpha=alphas[i],
            hatch=hatches[i],
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "black"},
            edgecolor="black",
            linewidth=0.6,
            label=sex,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [COND_LABELS.get(c, c) for c in all_conds],
        rotation=35, ha="right", fontsize=7,
    )
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def make_overview_figure(res: dict):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "VEN Fast Lane Hypothesis — NASA OSD-618 Behavioral Signatures\n"
        "Factorial: Housing × GCRsim Radiation × Sex  |  n = 118 mice\n"
        "Solid bars = male  hatched bars = female  reference = GH Sham",
        fontweight="bold", fontsize=11,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.70, wspace=0.55)
    panels = [
        (gs[0, 0], "social_sociability", "Sociability (3-chamber)", False),
        (gs[0, 1], "social_memory", "Social Memory (3-chamber)", False),
        (gs[0, 2], "nor", "Novel Object Recognition", False),
        (gs[1, 0], "rawm", "Radial Arm Water Maze\n(errors ↓ = better)", True),
        (gs[1, 1], "open_field", "Open Field (locomotor ctrl)", False),
        (gs[1, 2], "balance", "Balance Beam (motor ctrl)", False),
    ]

    for pos, key, title, hiw in panels:
        ax = fig.add_subplot(pos)
        entry = res.get(key, {})
        plot_bar_bysex(
            ax,
            entry.get("summary", {}),
            entry.get("ylabel", ""),
            title,
            higher_is_worse=hiw,
        )

    out = FIGS / "fig_ven_spaceflight_overview.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


def make_deficit_heatmap(res: dict):
    """
    Z-score heatmap vs GH_sham reference, one panel per sex.
    Z = (condition_mean - ref_mean) / ref_SD; sign-flipped for error metrics.
    """
    keys = ["social_sociability", "social_memory", "nor",
              "rawm", "open_field", "balance"]
    labels = ["Sociability", "Social Memory", "NOR",
              "RAWM Errors", "Open Field", "Balance Beam"]
    flip = {"rawm"}   # higher = worse for error counts

    compare_conds = [c for c in ORDERED_CONDITIONS if c != REFERENCE_CONDITION]

    # Collect all sexes present in any result
    sexes = sorted({
        sex
        for v in res.values()
        for sex in v.get("summary", {}).keys()
    })

    fig, axes = plt.subplots(1, len(sexes), figsize=(7 * len(sexes), 5.5))
    if len(sexes) == 1:
        axes = [axes]

    fig.suptitle(
        "Behavioral Deficit Heatmap - NASA OSD-618\n"
        f"Z-score vs {REFERENCE_CONDITION} (Group Housed, Sham-irradiated)\n"
        "Blue = better than control  Red = worse than control",
        fontweight="bold", fontsize=11,
    )

    im = None
    for ax, sex in zip(axes, sexes):
        matrix = np.full((len(keys), len(compare_conds)), np.nan)

        for i, key in enumerate(keys):
            ref_d = res.get(key, {}).get("summary", {}).get(sex, {}).get(
                REFERENCE_CONDITION, {}
            )
            if not ref_d or ref_d["n"] < 2:
                continue
            ref_mean = ref_d["mean"]
            ref_sd   = max(float(np.std(ref_d["raw"], ddof=1)), 1e-6)

            for j, cond in enumerate(compare_conds):
                cond_d = res.get(key, {}).get("summary", {}).get(sex, {}).get(cond, {})
                if not cond_d:
                    continue
                z = (cond_d["mean"] - ref_mean) / ref_sd
                matrix[i, j] = -z if key in flip else z

        im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-3, vmax=3)
        ax.set_xticks(range(len(compare_conds)))
        ax.set_xticklabels(
            [COND_LABELS.get(c, c) for c in compare_conds],
            rotation=35, ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"Sex: {sex}", fontweight="bold", fontsize=10)

        for ii in range(len(keys)):
            for jj in range(len(compare_conds)):
                v = matrix[ii, jj]
                if not np.isnan(v):
                    ax.text(
                        jj, ii, f"{v:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > 1.8 else "black",
                    )

    if im is not None:
        plt.colorbar(im, ax=axes, label="Z-score (positive = better than control)",
                     shrink=0.75, pad=0.04)

    plt.tight_layout()
    out = FIGS / "fig_deficit_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out}")


 #Save results JSON
 
def save_results(results: dict):
    out = RES / "ven_spaceflight_results.json"
    clean: dict = {}
    for assay_key, v in results.items():
        clean[assay_key] = {
            "metric": v.get("metric"),
            "ylabel": v.get("ylabel"),
            "higher_is_worse": v.get("higher_is_worse", False),
            "summary": {},
        }
        for sex, cond_dict in v.get("summary", {}).items():
            clean[assay_key]["summary"][sex] = {}
            for cond, s in cond_dict.items():
                clean[assay_key]["summary"][sex][cond] = {
                    "mean": round(s["mean"], 6),
                    "sem": round(s["sem"],  6),
                    "n": s["n"],
                }
    with open(out, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Saved: {out}")


def main():
    print("VEN Fast Lane Hypothesis - OSD-618 Behavioral Analysis")

    print("\n[1/3] Loading ISA metadata from OSD-618_metadata_OSD-618-ISA.zip")
    try:
        meta = load_isa_metadata(DATA)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)

    print("[2/3] Running behavioral analyses")
    results: dict = {}
    analyses = [
        ("Social Approach", analyse_social),
        ("NOR", analyse_nor),
        ("RAWM", analyse_rawm),
        ("Open Field", analyse_of),
        ("Balance Beam", analyse_balance),
    ]

    skipped = []
    for name, fn in analyses:
        try:
            fn(meta, results)
        except FileNotFoundError as e:
            print(f"\n [SKIP] {name}: {e}")
            skipped.append(name)
        except (ValueError, RuntimeError) as e:
            print(f"\n [ERROR] {name}: {e}")
            skipped.append(name)

    if not results:
        print(
            "\n[FATAL] No analyses completed.\n"
            "Check that behavioral CSV files are present in data/ "
            "and that animal IDs match the metadata."
        )
        sys.exit(1)

    if skipped:
        print(f"\n  Skipped (missing/invalid files): {skipped}")

    print("\n[3/3] Generating figures and saving results")
    make_overview_figure(results)
    make_deficit_heatmap(results)
    save_results(results)

    print("Complete.")


if __name__ == "__main__":
    main()
