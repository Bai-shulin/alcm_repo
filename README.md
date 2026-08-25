[中文说明](README.zh-CN.md) | [English](README.en.md)

# AI-Guided Dataset Construction and Active Learning for Carrier Mobility

This repository provides a modular workflow for **carrier mobility data construction, feature engineering, active learning, and candidate screening** in materials research.

The methodological workflow is designed to be **extensible across material systems**. The current repository uses **Bi₂Te₃-based thermoelectric materials as a case study** to demonstrate the complete pipeline, rather than restricting the framework to Bi₂Te₃ itself.

**dataset construction → feature engineering → active learning → candidate screening**

The workflow is organized into four connected stages. Dataset construction can start from curated records or user-specified literature; when literature is used, the Agent reads the supplied papers, identifies sample- and condition-level mobility data, and converts them into structured records for downstream modeling. Feature engineering, iterative model refinement, acquisition, and candidate ranking form the subsequent stages.

## Framework at a glance

| Module | General role | Bi₂Te₃ case study in this repository |
|---|---|---|
| **Dataset construction** | Build structured carrier mobility datasets from curated records and user-specified literature | Assemble and extend the Bi₂Te₃-based carrier mobility dataset |
| **Feature engineering** | Convert composition and experimental conditions into machine-learning descriptors | Generate elemental/statistical descriptors for Bi₂Te₃-based samples |
| **Active learning** | Iteratively identify informative features and high-value samples/candidates | Learn from the Bi₂Te₃ carrier mobility dataset using LightGBM, GPR, and Expected Improvement |
| **Candidate screening** | Rank unexplored compositions or modifications for follow-up | Prioritize candidate dopants/compositions for the Bi₂Te₃ demonstration |

## Bi₂Te₃ case study

<p align="center">
  <img src="assets/project_overview.png" alt="Bi2Te3 case study: literature-driven dataset construction, active learning, and carrier mobility screening" width="920">
</p>

The figure above illustrates the application implemented in the accompanying study. In this case, literature-derived carrier mobility data are organized into a structured dataset, elemental and condition descriptors are constructed, active learning is used to refine the model and candidate space, and promising dopant directions are evaluated for Bi₂Te₃-based materials.

The **general workflow** and the **materials-specific configuration** are intentionally separated. The same pipeline can be adapted to another material family by replacing the input dataset/literature set and, where needed, updating the extraction schema, host-element definition, descriptors, and candidate space.

## Workflow

### 1. Dataset construction

A structured carrier mobility dataset provides the common interface between literature/data ingestion and downstream machine learning. In the included case study, this dataset is `data/Electricity_complete.csv`.

The repository supports two data sources within the same module:

- curated or previously assembled structured records;
- records extracted from scientific papers explicitly supplied by the user.

For literature-derived data, the Agent can ingest local PDF/HTML files, direct URLs, DOI identifiers, or JSON/CSV/TXT manifests. It reads text and tables, identifies carrier-mobility measurements, extracts sample- and condition-level records, maps them to the configured dataset schema, and stores provenance separately.

A single paper may generate multiple rows corresponding to different compositions, temperatures, carrier concentrations, crystal forms, or measurement directions. Literature discovery is outside this module: the Agent processes the literature set provided to it.

### 2. Feature engineering

Material compositions and experimental conditions are converted into numerical descriptors. The current case-study implementation uses 30 elemental properties, four aggregation statistics, and temperature-weighted counterparts, producing 240 features for model training.

The feature-construction code is composition based rather than tied to a single chemical formula, so it can be reused for other material systems when the required elemental properties and input columns are available.

### 3. Iterative active learning

At each iteration, the workflow:

1. ranks features with LightGBM;
2. removes strongly correlated descriptors;
3. retains the most informative features;
4. fits a Gaussian Process Regression model with a Matérn kernel;
5. evaluates predictive performance on a fixed test set;
6. selects new samples using Expected Improvement.

These modeling steps are independent of the Bi₂Te₃ chemistry. Material-specific assumptions enter through the dataset, descriptors, candidate pool, and optional host-element definition used for dopant aggregation.

### 4. Candidate screening

After iterative model refinement, the trained workflow predicts candidate samples and ranks promising directions for follow-up. In the included Bi₂Te₃ case study, this stage is used to compare candidate dopants/compositions associated with improved carrier mobility.

For another host system, the candidate pool and `host_elements` setting can be changed without altering the core active-learning loop.

## Repository structure

```text
.
├── assets/
│   └── project_overview.png          # Bi2Te3 case-study figure
├── data/
│   ├── Electricity_complete.csv      # case-study carrier mobility dataset
│   ├── FVectors_MMS.csv              # precomputed case-study features
│   ├── N_type_carrier.csv            # target values
│   ├── candidate.csv                  # case-study candidate pool
│   └── dataset_sources.csv
├── literature/
│   ├── AGENT_PIPELINE.md
│   ├── extraction_schema.json        # Bi2Te3 case-study extraction config
│   ├── extraction_schema.template.json
│   ├── agent_inputs.example.json
│   └── sources.csv
├── notebooks/
│   ├── 01_construct_features.ipynb
│   └── 02_active_learning_gpr.ipynb
├── src/
│   ├── features.py
│   ├── active_learning.py
│   └── literature_mining/
│       ├── agent.py
│       ├── pipeline.py
│       └── cli.py
├── tests/
├── literature_pipeline.py
├── requirements.txt
├── requirements-literature.txt
├── DATASET_SOURCES.md
├── references.bib
└── README.md
```

## Installation

For the complete workflow:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For literature-to-dataset extraction only:

```bash
pip install -r requirements-literature.txt
```

## Quick start: reproduce the Bi₂Te₃ case study

### Build features

Run:

```text
notebooks/01_construct_features.ipynb
```

This creates the feature representation used by the active-learning workflow. If precomputed `data/FVectors_MMS.csv` and `data/N_type_carrier.csv` are available, this step can be skipped.

### Run active learning

Run:

```text
notebooks/02_active_learning_gpr.ipynb
```

The default case-study configuration uses 50 initial training samples, 50 fixed test samples, 20 Expected-Improvement selections per iteration, 10 retained features, a correlation cutoff of 0.95, `seed=2`, and `host_elements=("Bi", "Te")` for dopant-level aggregation.

### Extend the case-study dataset from supplied papers

```bash
python literature_pipeline.py doctor
python literature_pipeline.py agent-run \
  --papers literature/agent_inputs.example.json \
  --schema literature/extraction_schema.json \
  --base data/Electricity_complete.csv \
  --output data/Electricity_complete.agent.csv \
  --provenance data/Electricity_complete.agent.provenance.jsonl \
  --report literature/agent_report.json
```

The generated rows follow the schema of `data/Electricity_complete.csv`. Provenance such as DOI, page/table location, source evidence, extraction method, and confidence is stored separately so that extracted records remain traceable.

Detailed input formats and extension points are documented in [`literature/AGENT_PIPELINE.md`](literature/AGENT_PIPELINE.md).

## Adapting the workflow to another material system

The repository ships with Bi₂Te₃-oriented data and defaults because that is the demonstrated application. To use the workflow for another material family, the main changes are configuration/data changes rather than a rewrite of the pipeline:

1. provide a carrier mobility dataset using the required project fields, or define the corresponding schema mapping;
2. copy `literature/extraction_schema.template.json` and set material filters/aliases for the new system;
3. provide literature for that material family if using the literature Agent;
4. construct descriptors and a candidate pool appropriate to the new chemistry;
5. set `ActiveLearningConfig(host_elements=(...))` when dopant-level aggregation is required.

For example:

```python
from src.active_learning import ActiveLearningConfig

config = ActiveLearningConfig(host_elements=("Sn", "Se"))
```

This separation keeps **the reusable method** distinct from **the Bi₂Te₃ case-study inputs and scientific assumptions**.

## Dataset and provenance

The dataset included in this repository is the **Bi₂Te₃ case-study dataset**, assembled from published work on Bi₂Te₃-based and related thermoelectric systems. Its source metadata is maintained in:

- [`DATASET_SOURCES.md`](DATASET_SOURCES.md) — human-readable source list;
- [`references.bib`](references.bib) — BibTeX references;
- [`data/dataset_sources.csv`](data/dataset_sources.csv) — machine-readable source metadata.

These files document the included demonstration dataset; they do not define the material scope of the framework itself.

## Reproducibility and scientific definitions

The refactored implementation preserves several definitions and formulas from the original Bi₂Te₃ study so that its existing model inputs and results remain reproducible. Notes requiring scientific review are collected in [`VALIDATION_NOTES.md`](VALIDATION_NOTES.md) rather than silently changing the original definitions.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Contributors

**Shulin Bai** — Beihang University  
**Pengfei Zhang** — Beihang University

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for contributor information.

## Citation

If you use this code or the included dataset, please cite the associated research work. Citation metadata will be updated in this repository when the corresponding publication information is available.

## License

See [`LICENSE`](LICENSE).
