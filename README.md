# Running the code - Quick‑Start Guide
A fully‑worked, single‑file notebook that:

* connects to a **local PostgreSQL copy of the MIMIC‑III v1.4** database,
* recreates several analyses/figures from **Boag et al., 2022 (“EHR Safari: Data is Contextual”)**, and  
* demonstrates how to pull MIMIC‑III data into **PyHealth** style workflows.

---

## 1 . Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Operating system** | Linux/macOS/WSL recommended (PostgreSQL + Python ≥ 3.9) |
| **Access to raw MIMIC‑III CSVs** | You *must* (i) complete CITI training, (ii) sign the DUA, and (iii) request access on <https://physionet.org>. |
| **PostgreSQL ≥ 13** | Used to host the dataset locally. |
| **Python ≥ 3.9** | All code was tested on 3.10. |

---

## 2 . Installing the Python Environment

```bash
# 1. Clone (or copy) this repository
git clone <your‑fork‑or‑location>
cd <repo_root>

# 2. Create & activate a virtual‑env (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install required libraries
pip install --upgrade pip

# Core scientific stack
pip install numpy pandas matplotlib tqdm scipy xgboost lifelines

# Database / I/O
pip install psycopg sqlalchemy ipython ipykernel

# Optional (nice‑to‑have): pretty display, progress bars, etc.
pip install jupyterlab seaborn

# For PyHealth (if you also want to use the package)
pip install pyhealth
