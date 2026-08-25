# Feature implementation notes

This document records implementation details used in the feature-construction workflow.

## Fixed without changing the intended workflow

- Removed the hard-coded `Number = 324`; feature arrays now use the actual dataset length.
- Removed repeated imports and 30 near-identical feature-construction blocks.
- Added input validation, deterministic ordering, explicit paths, reusable functions,
  structured output folders, safer handling of zero feature-importance sums, and explicit
  exclusion of non-finite columns from the GPR-selected feature subset.
- Replaced the effectively unbounded `MAX_ITER = 99999` with a limit determined by the
  available active-learning pool.
- Saved iteration histories and final feature summaries explicitly.
- Split composition-level ranking from true dopant-element aggregation to avoid a
  misleading filename/interpretation.
- Added parity and residual plot generation for each iteration.

## Scientific definitions intentionally preserved for reproducibility

The following definitions are preserved from the original implementation:

1. **Legacy WAM denominator**  
   The original code computes
   `sum(composition_i * property_i) / sum(property_i)` over present elements.
   A conventional composition-weighted arithmetic mean would normally divide by
   `sum(composition_i)`. The refactor keeps the original expression.

2. **Legacy WSD center**  
   The original WSD uses the mean of `composition_i * property_i` as the center in the
   squared-deviation term. This is kept unchanged.

3. **Third ionization-energy descriptor**  
   In the original notebook, both the second and third ionization-energy descriptors use
   `ionenergies[2]`. A true third ionization energy would generally use `ionenergies[3]`.
   The refactor preserves `[2]` so existing feature matrices/results are not silently changed.

4. **Missing elemental properties**  
   Ionization-energy exceptions are represented as zero, matching the original code.
   This behavior is retained consistently in feature generation.

## Literature dataset provenance requiring author confirmation

1. **Categorical codebook is missing**
   `Electricity_complete.csv` uses `Crystal_form` codes 0/1 and `Direction` codes
   0/1/2, but their scientific meanings are not documented in the repository.
   The literature extractor therefore leaves these fields blank and requires
   review. Candidate mappings in `literature/extraction_schema.json` are disabled
   until the dataset owner confirms them.

2. **Legacy rows are not linked to individual sources**
   The bibliography documents the literature collection, but the 280 legacy data
   rows do not include a DOI, page, table/figure, or evidence field. The new
   pipeline keeps provenance for added rows in a separate generated CSV, but it
   cannot reconstruct legacy row-level provenance without author records or a
   manual audit.

3. **Plot-only values must not be guessed**
   When a paper discusses mobility but provides values only in figures, the
   extractor exports the relevant PDF pages and marks the paper for calibrated
   digitization. Those values are not automatically approved.

## Framework scope versus case-study defaults

The reusable workflow is not limited to Bi2Te3. The repository nevertheless keeps
Bi/Te-oriented defaults where they are required to reproduce the included case study.
These defaults are now explicit configuration points rather than hidden assumptions:

- literature formula filtering is controlled by `validation.required_elements` in the
  extraction schema;
- `literature/extraction_schema.json` is the Bi2Te3 case-study configuration, while
  `literature/extraction_schema.template.json` is a material-neutral starting point;
- dopant-level aggregation uses `ActiveLearningConfig.host_elements`, which defaults to
  `(\"Bi\", \"Te\")` for the included demonstration and can be changed for another host.
