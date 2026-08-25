from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from literature_mining.pipeline import (  # noqa: E402
    DATASET_COLUMNS,
    REVIEW_COLUMNS,
    CandidateRecord,
    SourceRecord,
    _is_safe_remote_url,
    _write_review_queue_preserving_decisions,
    build_dataset,
    extract_formula,
    extract_table_candidates,
    extract_text_candidates,
    normalize_doi,
    parse_scientific_number,
    write_csv,
)


class NormalizationTests(unittest.TestCase):
    def test_doi_url_is_normalized(self) -> None:
        self.assertEqual(
            normalize_doi("https://doi.org/10.3390/APP8050735."),
            "10.3390/app8050735",
        )

    def test_unicode_scientific_notation(self) -> None:
        self.assertAlmostEqual(parse_scientific_number("2.5 × 10−3"), 0.0025)

    def test_bismuth_telluride_formula_is_extracted(self) -> None:
        text = "Sample (Bi0.9Sb0.1)2(Te0.85Se0.15)3 was consolidated."
        self.assertEqual(extract_formula(text), "(Bi0.9Sb0.1)2(Te0.85Se0.15)3")

    def test_formula_filter_is_material_system_configurable(self) -> None:
        schema = {"validation": {"required_elements": ["Ga", "As"]}}
        self.assertEqual(extract_formula("The GaAs sample was measured.", schema), "GaAs")
        self.assertEqual(extract_formula("The Bi2Te3 sample was measured.", schema), "")

    def test_private_download_destination_is_rejected(self) -> None:
        safe, reason = _is_safe_remote_url("http://127.0.0.1/paper.pdf")
        self.assertFalse(safe)
        self.assertIn("non-public", reason)


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(
            (PROJECT_ROOT / "literature" / "extraction_schema.json").read_text(
                encoding="utf-8"
            )
        )

    def test_table_candidate_preserves_evidence_and_units(self) -> None:
        rows = [
            ["Composition", "Temperature (K)", "Hall mobility (cm2 V-1 s-1)"],
            ["Bi2Te2.7Se0.3", "300", "125.5"],
        ]
        source = SourceRecord("example", "10.1000/example", "Example paper")

        candidates = extract_table_candidates(rows, source, self.schema, page_number=4)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.Crystal_structure, "Bi2Te2.7Se0.3")
        self.assertEqual(candidate.temperature, "300")
        self.assertEqual(candidate.N_type_carrier, "125.5")
        self.assertEqual(candidate.page, "4")
        self.assertEqual(candidate.evidence_type, "table")
        self.assertEqual(candidate.review_status, "needs_review")

    def test_scaled_square_metre_unit_is_converted(self) -> None:
        rows = [
            ["Composition", "Temperature (K)", "Mobility (10^-4 m2 V-1 s-1)"],
            ["Bi2Te3", "300", "150"],
        ]
        source = SourceRecord("example")

        candidate = extract_table_candidates(rows, source, self.schema, 1)[0]

        self.assertEqual(candidate.N_type_carrier, "150")

    def test_parallel_values_are_paired_with_formulas(self) -> None:
        text = (
            "Using the Hall effect to determine carrier concentration and mobility, "
            "the results are 9.34, 7.18 (×10^19 cm-3) and 58.3, 56.4 "
            "(cm2/(V s)) for Lu0.1Bi1.9Te3, Lu0.1Bi1.9Te2.8Se0.2, respectively."
        )
        source = SourceRecord("parallel", title="Parallel lists")

        candidates = extract_text_candidates(text, source, self.schema, 5)

        paired = [
            (candidate.Crystal_structure, candidate.N_type_carrier)
            for candidate in candidates
            if candidate.evidence_type == "parallel_text_list"
        ]
        self.assertEqual(
            paired,
            [("Lu0.1Bi1.9Te3", "58.3"), ("Lu0.1Bi1.9Te2.8Se0.2", "56.4")],
        )


class DatasetBuildTests(unittest.TestCase):
    def test_refresh_preserves_reviewed_edits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            queue = Path(temporary) / "review.csv"
            candidate = CandidateRecord(
                source_id="paper",
                doi="10.1000/paper",
                title="Paper",
                Crystal_structure="Bi2Te3",
                temperature="300",
                N_type_carrier="100",
                Crystal_form="",
                Direction="",
                evidence="extracted",
            )
            old_row = candidate.as_row()
            old_row["N_type_carrier"] = "101.5"
            old_row["Crystal_form"] = "0"
            old_row["Direction"] = "0"
            old_row["review_status"] = "approved"
            old_row["reviewer_notes"] = "checked against Table 2"
            write_csv(queue, REVIEW_COLUMNS, [old_row])

            _write_review_queue_preserving_decisions(queue, [candidate])

            with queue.open(encoding="utf-8-sig", newline="") as handle:
                refreshed = list(csv.DictReader(handle))
            self.assertEqual(refreshed[0]["N_type_carrier"], "101.5")
            self.assertEqual(refreshed[0]["review_status"], "approved")
            self.assertEqual(refreshed[0]["reviewer_notes"], "checked against Table 2")

    def test_only_approved_non_duplicate_rows_are_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.csv"
            review = root / "review.csv"
            output = root / "output.csv"
            provenance = root / "provenance.csv"

            write_csv(
                base,
                DATASET_COLUMNS,
                [
                    {
                        "Crystal_structure": "Bi2Te3",
                        "temperature": "300",
                        "N_type_carrier": "100",
                        "Crystal_form": "0",
                        "Direction": "0",
                    }
                ],
            )
            approved_new = CandidateRecord(
                source_id="new-paper",
                doi="10.1000/new",
                title="New",
                Crystal_structure="Bi2Te2.7Se0.3",
                temperature="300",
                N_type_carrier="125",
                Crystal_form="0",
                Direction="0",
                page="5",
                evidence_type="table",
                evidence="verified row",
                review_status="approved",
            )
            approved_duplicate = CandidateRecord(
                source_id="duplicate",
                doi="10.1000/duplicate",
                title="Duplicate",
                Crystal_structure="Bi2Te3",
                temperature="300",
                N_type_carrier="100",
                Crystal_form="0",
                Direction="0",
                review_status="approved",
            )
            rejected = CandidateRecord(
                source_id="rejected",
                doi="10.1000/rejected",
                title="Rejected",
                Crystal_structure="Bi2Te3Cu0.01",
                temperature="300",
                N_type_carrier="99",
                Crystal_form="0",
                Direction="0",
                review_status="rejected",
            )
            write_csv(
                review,
                REVIEW_COLUMNS,
                [
                    approved_new.as_row(),
                    approved_duplicate.as_row(),
                    rejected.as_row(),
                ],
            )

            result = build_dataset(
                base,
                review,
                PROJECT_ROOT / "literature" / "extraction_schema.json",
                output,
                provenance,
            )

            self.assertEqual(result["added_rows"], 1)
            self.assertEqual(result["duplicate_rows"], 1)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            with provenance.open(encoding="utf-8-sig", newline="") as handle:
                provenance_rows = list(csv.DictReader(handle))
            self.assertEqual(len(provenance_rows), 1)
            self.assertEqual(provenance_rows[0]["doi"], "10.1000/new")


if __name__ == "__main__":
    unittest.main()
