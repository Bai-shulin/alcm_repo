from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from literature_mining.agent import load_paper_inputs, run_agent_pipeline  # noqa: E402
from literature_mining.pipeline import DATASET_COLUMNS, write_csv  # noqa: E402


class AgentInputTests(unittest.TestCase):
    def test_manifest_normalizes_doi_without_discovery(self) -> None:
        inputs = load_paper_inputs([{"source_id": "p", "doi": "doi:10.1000/EXAMPLE"}])
        self.assertEqual(inputs[0].doi, "10.1000/example")
        self.assertEqual(inputs[0].stable_id, "p")

    def test_html_document_generates_existing_dataset_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.csv"
            html = root / "paper.html"
            output = root / "agent.csv"
            provenance = root / "agent.jsonl"
            report = root / "report.json"
            cache = root / "cache"
            write_csv(
                base,
                DATASET_COLUMNS,
                [{
                    "Crystal_structure": "Bi2Te3",
                    "temperature": "300",
                    "N_type_carrier": "100",
                    "Crystal_form": "1",
                    "Direction": "0",
                }],
            )
            html.write_text(
                "<html><body><table>"
                "<tr><th>Composition</th><th>Temperature (K)</th>"
                "<th>Hall mobility (cm2 V-1 s-1)</th></tr>"
                "<tr><td>Bi2Te2.7Se0.3</td><td>300</td><td>125.5</td></tr>"
                "</table></body></html>",
                encoding="utf-8",
            )
            result = run_agent_pipeline(
                [str(html)], base, output, provenance, report, cache
            )
            self.assertEqual(result["added_rows"], 1)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]["Crystal_structure"], "Bi2Te2.7Se0.3")
            self.assertEqual(rows[-1]["N_type_carrier"], "125.5")
            self.assertEqual(list(rows[-1]), DATASET_COLUMNS)
            provenance_rows = [json.loads(line) for line in provenance.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(provenance_rows[0]["page"], "1")
            self.assertEqual(provenance_rows[0]["extraction_method"], "table")


if __name__ == "__main__":
    unittest.main()