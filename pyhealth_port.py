"""
------------------------------------------------------------------------
Re‑implementation of “single_file_port_to_pyhealth.py” using PyHealth
Author : <your name>
Date   : 2025‑04‑19
------------------------------------------------------------------------
PyHealth abstracts MIMIC‑III tables into Pandas DataFrames that live in
`dataset.tables`.  This script shows how to:

  •  Pull vital signs, labs, notes, admissions, ICU stays
  •  Reproduce Section 5 figures from Boag et al. (2022)
  •  Inspect data‑quality quirks without writing raw SQL

------------------------------------------------------------------------
Setup
------------------------------------------------------------------------
1)  pip install pyhealth[full] matplotlib tqdm
2)  export MIMIC3_DIR=/path/to/MIMIC-III/
    (directory that contains the canonical CSVs, e.g. ADMISSIONS.csv)
------------------------------------------------------------------------
"""
# %% imports
import os, warnings, datetime as dt
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from pyhealth.datasets import MIMIC3Dataset     # <-- high‑level loader

plt.style.use("ggplot")
warnings.filterwarnings("ignore")

# %% ------------------------------------------------------------------
# Load MIMIC‑III via PyHealth
# ---------------------------------------------------------------------
root = os.getenv("MIMIC3_DIR", "./mimic-iii-root")     # fallback path
dataset = MIMIC3Dataset(
    root=root,
    tables=[            # only what we need; speeds up IO
        "PATIENTS",
        "ADMISSIONS",
        "ICUSTAYS",
        "CHARTEVENTS",
        "LABEVENTS",
        "NOTEEVENTS",
        "D_ICD_DIAGNOSES",
        "DIAGNOSES_ICD",
    ],
    dev=False           # set True to load 10 % sample during debugging
)

patients     = dataset.tables["PATIENTS"]
admissions   = dataset.tables["ADMISSIONS"]
icustays     = dataset.tables["ICUSTAYS"]
chartevents  = dataset.tables["CHARTEVENTS"]
labevents    = dataset.tables["LABEVENTS"]
noteevents   = dataset.tables["NOTEEVENTS"]

# Helper: convert DOB → age at arbitrary time -------------------------
def compute_age(row_time, dob):
    return (row_time - dob).days / 365.25

# %% ==================================================================
# 5.1  “Surprising Distribution of Heart Rates”  (Figures 2 & 3)
# =====================================================================
HR_ITEMIDS = [211, 220045, 5972]               # adult, meta, fetal
hr_df = chartevents.loc[
    chartevents["itemid"].isin(HR_ITEMIDS),
    ["subject_id", "hadm_id", "charttime", "valuenum"]
].rename(columns={"valuenum": "heart_rate"})

# merge DOB to compute age
hr_df = (
    hr_df
    .merge(patients[["subject_id", "dob"]], on="subject_id", how="left")
    .assign(age=lambda d: d.apply(lambda r: compute_age(r.charttime, r.dob), axis=1))
    .query("heart_rate>0 & heart_rate<=250")
)

# ---- Figure 2 : Histogram -------------------------------------------
plt.figure(figsize=(6,3))
plt.hist(hr_df["heart_rate"], bins=np.arange(0,260,3), color="#2678b2")
plt.xlabel("beats per minute (bpm)"); plt.ylabel("number of measurements")
plt.title("Heart‑rate distribution (all itemids)"); plt.tight_layout()
plt.show()

# ---- Figure 3 : HR vs Age scatter -----------------------------------
plt.figure(figsize=(6,4))
plt.scatter(hr_df["age"], hr_df["heart_rate"], s=2, alpha=0.02, color="#2600ff")
plt.xlabel("Patient Age (years)"); plt.ylabel("Heart Rate (bpm)")
plt.title("Heart rate versus age"); plt.xlim(-1, 320); plt.ylim(0,260)
plt.tight_layout(); plt.show()

# %% ==================================================================
# 5.2  “Inconsistent Timestamps”
# =====================================================================
icu_adm = (
    icustays[["hadm_id","icustay_id","intime","outtime"]]
    .merge(admissions[["hadm_id","admittime","dischtime"]], on="hadm_id")
)

conds = {
    "intime_before_admit": icu_adm["intime"] < icu_adm["admittime"],
    "outtime_after_disch" : icu_adm["outtime"] > icu_adm["dischtime"],
}
icu_adm = icu_adm.assign(**conds)

total = len(icu_adm)
print(f"Total ICU stays: {total:,}")
for k in conds:
    n = icu_adm[k].sum()
    print(f"{k.replace('_',' ')}: {n:,}  ({n/total*100:.1f} %)")

# %% ==================================================================
# 5.3  “Lab Values Vary by Time of Day”  (Figure 5)
# =====================================================================
WBC_ITEMIDS = [51300, 51301]   # manual diff + auto diff
wbc = labevents.loc[
    labevents["itemid"].isin(WBC_ITEMIDS),
    ["charttime","valuenum"]
].rename(columns={"valuenum":"wbc"})
wbc = wbc.assign(hour=wbc["charttime"].dt.hour,
                 minute=wbc["charttime"].dt.minute)
wbc["is_abn"] = ~wbc["wbc"].between(4.5, 11.0)

# aggregate per half‑hour bucket
wbc["bucket"] = wbc["hour"] + (wbc["minute"]>=30)*0.5
agg = (
    wbc.groupby("bucket")["is_abn"]
    .agg(["mean","count"]).reset_index()
    .rename(columns={"mean":"pct_abn","count":"n"})
)

plt.figure(figsize=(7,4))
plt.plot(agg["bucket"], agg["pct_abn"], lw=2, color="#0044cc")
plt.axvspan(4,8, color="orange", alpha=.15, label="04:00–08:00")
plt.ylim(0,1); plt.xticks(range(0,24,2)); plt.legend()
plt.ylabel("Fraction abnormal"); plt.xlabel("Hour of day")
plt.title("WBC abnormal fraction by time of day"); plt.tight_layout(); plt.show()

# %% ==================================================================
# 5.4  “Multiple Copies of Provider Notes”
# =====================================================================
# Each autosave draft shares the same charttime but different storetime.
draft_stats = (
    noteevents
    .groupby(["subject_id","hadm_id","charttime"])
    .size()
    .reset_index(name="versions")
)

n_total   = len(draft_stats)
multi     = (draft_stats["versions"]>1).sum()
max_ver   = draft_stats["versions"].max()
print(f"Distinct notes (charttime key): {n_total:,}")
print(f"With ≥2 versions (autosaves):   {multi:,}")
print(f"Maximum versions seen:          {max_ver}")

# %% ==================================================================
# 5.5  “Missing Death‑Date Collection”  (Kaplan–Meier example)
# =====================================================================
from lifelines import KaplanMeierFitter

post = (
    admissions.query("deathtime.notna() & dischtime.notna()")
    .assign(
        t=lambda d: (d["deathtime"]-d["dischtime"]).dt.days,
        event=1
    )
    .merge(patients[["subject_id","dod","gender","ethnicity"]], on="subject_id")
)
post["race"] = post["ethnicity"].str.extract(r"(BLACK|WHITE|ASIAN|HISPANIC)",
                                             expand=False).fillna("OTHER")

kmf = KaplanMeierFitter()
plt.figure(figsize=(7,4))
for race,color in [("WHITE","orange"),("BLACK","#348fc2")]:
    subset = post.loc[post["race"]==race, ["t","event"]].dropna()
    kmf.fit(durations=subset["t"], event_observed=subset["event"], label=race)
    kmf.plot_survival_function(ci_show=False, color=color)
plt.xlabel("Days since discharge"); plt.ylabel("Survival probability")
plt.title("Post‑discharge survival by race"); plt.tight_layout(); plt.show()

# %% ==================================================================
# 6.1  Admission Diagnosis vs ICD Codes
# =====================================================================
adm_diag = admissions[["hadm_id","diagnosis"]]
icd9 = (
    dataset.tables["DIAGNOSES_ICD"]
    .merge(dataset.tables["D_ICD_DIAGNOSES"][["icd9_code","long_title"]],
           on="icd9_code")
    .groupby("hadm_id")["long_title"].apply(lambda x: "; ".join(sorted(set(x))))
    .reset_index(name="icd_titles")
)
compare_df = adm_diag.merge(icd9, on="hadm_id", how="inner")
display(compare_df.sample(5))

# %% ==================================================================
# 6.2  Variation in Data Entry Proxy  (hourly box‑plot)
# =====================================================================
proxy = (
    chartevents.query("itemid==220045 & valuenum.between(0,100)")
    [["charttime","valuenum"]].rename(columns={"valuenum":"proxy"})
)
proxy["shift"] = pd.cut(proxy["charttime"].dt.hour,
                        bins=[-1,7,15,23],
                        labels=["Night","Day","Evening"])
proxy.boxplot(column="proxy", by="shift"); plt.suptitle("")
plt.title("Proxy distribution by nursing shift"); plt.tight_layout(); plt.show()
