# Input-driven literature extraction Agent

This module is the literature-ingestion path of the repository's **dataset construction stage**. It processes only references explicitly supplied by the user; it does not search for or recommend papers.

The repository's default configuration demonstrates the workflow on Bi₂Te₃-based carrier-mobility literature, while the extraction filter can be changed through `--schema` for another material family.

## Inputs

`--papers` may point to a JSON, CSV, or text manifest, or may be one PDF/HTML path, URL, or DOI. JSON accepts either a list or `{ "papers": [...] }`:

```json
[
  {"source_id": "paper_a", "path": "papers/paper_a.pdf", "doi": "10.xxxx/aaaa"},
  {"source_id": "paper_b", "url": "https://example.org/paper.pdf"},
  {"source_id": "paper_c", "doi": "10.xxxx/cccc"}
]
```

A CSV manifest may use `source_id`, `path`/`file`/`pdf`/`url`/`doi`, `title`, and `notes` columns. A DOI is resolved only as the supplied document identifier; no bibliography-discovery step is performed.

## Extraction schema

`--schema` controls material filters, field aliases, categorical mappings, and validation ranges.

- `literature/extraction_schema.json` is the **Bi₂Te₃ case-study configuration**.
- `literature/extraction_schema.template.json` is a material-neutral starting point.
- `validation.required_elements` can be set to a host chemistry such as `["Bi", "Te"]`, or left empty to disable host-element filtering during formula extraction/validation.

## Run the included case study

```powershell
python literature_pipeline.py agent-run `
  --papers literature/agent_inputs.example.json `
  --schema literature/extraction_schema.json `
  --base data/Electricity_complete.csv `
  --output data/Electricity_complete.agent.csv `
  --provenance data/Electricity_complete.agent.provenance.jsonl `
  --report literature/agent_report.json
```

The default output uses the five carrier-mobility columns in `data/Electricity_complete.csv`:

`Crystal_structure, temperature, N_type_carrier, Crystal_form, Direction`

The Agent can emit multiple rows for one paper. It reads PDF text and detected tables, handles HTML tables, detects mobility-bearing passages, and keeps rows with missing optional fields rather than turning the workflow into a heavy validation framework. Papers with no target mention or no numeric candidate are reported and skipped.

`*.provenance.jsonl` contains DOI, page, table number when available, source text, extraction method, confidence, validation notes, and the mapped dataset row. The report contains one status/reason per supplied paper.

## Using another material system

Create a schema from the template and pass it to `agent-run`:

```powershell
python literature_pipeline.py agent-run `
  --papers papers/other_system.json `
  --schema literature/other_system_schema.json `
  --base data/other_mobility_dataset.csv `
  --output data/other_mobility_dataset.agent.csv
```

The document loader, relevance assessment, text/table extraction, provenance capture, and dataset assembly are reusable. Material-specific behavior is concentrated in the supplied data/schema and downstream descriptor/candidate definitions.

## Extension point

`PaperJudge` in `src/literature_mining/agent.py` is a small interface. A larger Agent workflow can replace `HeuristicPaperJudge` with an LLM-backed judge while leaving document loading, extraction, schema mapping, and dataset construction unchanged. No model provider is hard-coded into the pipeline.
