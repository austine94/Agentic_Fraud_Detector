# Agentic Fraud Detector

This repository contains the code needed to reproduce the PaySim fraud
monitoring demo. The raw dataset and the generated SQLite databases are not
stored in Git because the two main databases are approximately 3 GB in total.
They can be rebuilt locally from the public PaySim CSV by following the steps
below.

Run every command from the repository root.

## 1. Prepare Python

Python 3.10 or newer is recommended. Create an isolated environment and
install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install numpy==2.0.2 scikit-learn==1.6.1
```

The data-building steps use Python's standard library, except for the frozen
IID split builder, which uses NumPy.

## 2. Download PaySim

Download **Synthetic Financial Datasets For Fraud Detection** from the
canonical [PaySim dataset page on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
The required file is:

```text
PS_20174392719_1491204439457_log.csv
```

Create a `data` directory and place the unmodified CSV at:

```text
data/PS_20174392719_1491204439457_log.csv
```

Alternatively, users with the
[Kaggle CLI](https://github.com/Kaggle/kaggle-api) configured can download and
extract it directly:

```bash
mkdir -p data
kaggle datasets download -d ealaxi/paysim1 -p data --unzip
```

Validate the file before building anything:

```bash
python -m src.data data/PS_20174392719_1491204439457_log.csv
```

The expected source fingerprint is:

```text
rows:   6,362,620
SHA256: 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
```

If the checksum differs, do not assume the generated databases will reproduce
the published setup.

## 3. Build the main case database

Build `data/evidence.sqlite` from the CSV:

```bash
python -m experiments.build_cases_store
```

This database contains all 6,362,620 PaySim rows with stable `case_id` values
and the chronological indexes used by the evidence tools. The resulting file
is approximately 1.1 GB.

## 4. Build the pre-authorisation behaviour database

Build `data/preauth_behavior.sqlite` one restartable stage at a time, in the
following order:

```bash
python -m experiments.build_preauth_behavior_stage init
python -m experiments.build_preauth_behavior_stage origin
python -m experiments.build_preauth_behavior_stage destination
python -m experiments.build_preauth_behavior_stage pair
python -m experiments.build_preauth_behavior_stage network
python -m experiments.build_preauth_behavior_stage transfer
python -m experiments.build_preauth_behavior_stage join
python -m experiments.build_preauth_behavior_stage verify
```

The stages calculate account, pair, and destination-network histories using
only transactions from strictly earlier PaySim steps. Same-step rows and
post-transaction balances are excluded from the behavioural profiles. The
finished database is approximately 1.7 GB, and verification should report:

```text
profiles=6362620
```

## 5. Build the frozen split assignment

The experiments also expect the smaller deterministic split database at
`data/iid_split.sqlite`. Generate it from `evidence.sqlite` with:

```bash
python -m experiments.build_iid_split
```

This creates the stratified 50% train / 10% validation / 40% test assignment
using seed `20260925`. The resulting file is approximately 94 MB.

## Generated files

After setup, the local `data` directory should contain:

```text
data/
├── PS_20174392719_1491204439457_log.csv  # downloaded source, about 471 MB
├── evidence.sqlite                       # generated, about 1.1 GB
├── preauth_behavior.sqlite               # generated, about 1.7 GB
└── iid_split.sqlite                      # generated, about 94 MB
```

Allow several additional gigabytes of free space while SQLite builds its
indexes and intermediate tables. The builders deliberately refuse to
overwrite existing output files. For a clean rebuild, first rename or remove
the relevant generated database and then rerun its commands.

None of these files should be committed. They are reproducible from the
downloaded PaySim CSV and are covered by this repository's `.gitignore`.

## Quick check

Once the databases exist, run the test suite from the repository root:

```bash
python -m pip install pytest
python -m pytest
```
