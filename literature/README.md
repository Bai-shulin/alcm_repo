# Literature-driven dataset construction

This directory contains the literature-processing component of the repository's **dataset construction module**. It converts user-specified scientific papers into structured carrier-mobility records that can enter the same downstream feature-engineering and active-learning workflow as curated data.

The component is not specific to Bi₂Te₃ at the workflow level. The repository ships with a **Bi₂Te₃ case-study schema** because that is the demonstrated application; users can supply another extraction schema for a different material family.

## Role in the full workflow

```text
curated data ───────────────┐
                            ├─> structured mobility dataset
user-specified literature ─>│       ↓
        Agent extraction ───┘   feature engineering
                                    ↓
                               active learning
                                    ↓
                             candidate screening
```

The literature Agent is therefore one route into dataset construction, not a separate or auxiliary research workflow.

## Agent workflow

The Agent accepts local PDF/HTML files, direct URLs, DOI identifiers, or JSON/CSV/TXT manifests. It operates on supplied references rather than performing literature discovery.

```text
user-specified papers
        ↓
document loading / parsing
        ↓
relevance and data detection
        ↓
sample-level extraction
        ↓
schema mapping
        ↓
carrier-mobility dataset + provenance
```

Run the included Bi₂Te₃ case-study configuration from the repository root:

```powershell
python -m pip install -r requirements-literature.txt
python literature_pipeline.py doctor

python literature_pipeline.py agent-run `
  --papers literature/agent_inputs.example.json `
  --schema literature/extraction_schema.json `
  --base data/Electricity_complete.csv `
  --output data/Electricity_complete.agent.csv `
  --provenance data/Electricity_complete.agent.provenance.jsonl `
  --report literature/agent_report.json
```

A paper may produce multiple rows when it reports different compositions, temperatures, carrier concentrations, crystal forms, or measurement directions. Provenance is written separately so each extracted row can retain DOI, page/table information, evidence text, extraction method, and confidence.

## Adapting extraction to another material family

`literature/extraction_schema.json` contains the configuration used for the Bi₂Te₃ demonstration. To adapt the extractor:

1. copy `literature/extraction_schema.template.json`;
2. set `target_description` for the new task;
3. update column aliases if the target dataset uses different terminology;
4. set `validation.required_elements` when extraction should be restricted to a host chemistry, or leave it empty for no host-element filter;
5. pass the new file through `--schema`.

Example:

```powershell
python literature_pipeline.py agent-run `
  --papers papers/my_materials.json `
  --schema literature/my_material_schema.json `
  --base data/my_mobility_dataset.csv `
  --output data/my_mobility_dataset.agent.csv
```

The current code writes the five carrier-mobility fields used by this repository (`Crystal_structure`, `temperature`, `N_type_carrier`, `Crystal_form`, `Direction`). Extending the output to a different property schema can be done at the schema-mapping/dataset-builder layer without changing document loading or paper acquisition.

For accepted manifest formats, provenance fields, and the `PaperJudge` extension interface, see [`AGENT_PIPELINE.md`](AGENT_PIPELINE.md).

## Review-queue workflow

A review-queue path is also retained for cases where extracted candidates should be manually approved before dataset assembly.

### Configure sources

Edit `literature/sources.csv`. Supported fields include `source_id`, `doi`, `title`, `pdf_url`, `local_pdf`, `enabled`, and `notes`.

### Extract candidates

```powershell
python literature_pipeline.py run --email your_contact_email@example.com
```

Typical outputs include `fetch_report.csv`, `extraction_report.csv`, `review_queue.csv`, and pages requiring manual figure digitization.

### Review and build

After reviewing candidate rows, generate a dataset with:

```powershell
python literature_pipeline.py build
```

This produces generated dataset/provenance files without overwriting the original dataset unless `--in-place` is explicitly requested.

## Extraction scope

The built-in parser handles machine-readable PDF/HTML text and recognizable tables. Values available only from plotted curves are flagged for manual digitization rather than inferred without calibrated axes.
