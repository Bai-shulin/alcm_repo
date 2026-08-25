"""Command-line interface for the literature mining pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
import re

from .agent import run_agent_pipeline
from .pipeline import build_dataset, extract_pdfs, fetch_sources


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m literature_mining",
        description=(
            "Fetch open-access papers, extract carrier-mobility candidates, "
            "and build a reviewed dataset."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=str(_project_root()),
        help="Project root (defaults to the repository containing this module).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check required Python packages and files.")
    doctor.add_argument("--schema", default="literature/extraction_schema.json")
    doctor.add_argument("--sources", default="literature/sources.csv")

    fetch = subparsers.add_parser("fetch", help="Download only openly accessible PDFs.")
    fetch.add_argument("--sources", default="literature/sources.csv")
    fetch.add_argument("--pdf-dir", default="literature/pdfs")
    fetch.add_argument("--report", default="literature/fetch_report.csv")
    fetch.add_argument("--email", default="", help="Contact email sent to scholarly APIs.")
    fetch.add_argument("--limit", type=int, default=None)

    extract = subparsers.add_parser("extract", help="Extract review candidates from local PDFs.")
    extract.add_argument("--sources", default="literature/sources.csv")
    extract.add_argument("--schema", default="literature/extraction_schema.json")
    extract.add_argument("--pdf-dir", default="literature/pdfs")
    extract.add_argument("--review", default="literature/review_queue.csv")
    extract.add_argument("--report", default="literature/extraction_report.csv")
    extract.add_argument("--figure-dir", default="literature/figures")
    extract.add_argument("--limit", type=int, default=None)

    run = subparsers.add_parser("run", help="Fetch PDFs and extract a review queue.")
    run.add_argument("--sources", default="literature/sources.csv")
    run.add_argument("--schema", default="literature/extraction_schema.json")
    run.add_argument("--pdf-dir", default="literature/pdfs")
    run.add_argument("--email", default="")
    run.add_argument("--limit", type=int, default=None)

    agent_run = subparsers.add_parser(
        "agent-run",
        help="Process only user-supplied PDFs, URLs, DOIs, or a paper manifest.",
    )
    agent_run.add_argument(
        "--papers",
        required=True,
        help="A PDF/HTML path, URL, DOI, or CSV/JSON/TXT manifest of supplied papers.",
    )
    agent_run.add_argument(
        "--schema",
        default="literature/extraction_schema.json",
        help="Extraction schema/material filter (defaults to the included Bi2Te3 case study).",
    )
    agent_run.add_argument("--base", default="data/Electricity_complete.csv")
    agent_run.add_argument("--output", default="data/Electricity_complete.agent.csv")
    agent_run.add_argument(
        "--provenance", default="data/Electricity_complete.agent.provenance.jsonl"
    )
    agent_run.add_argument("--report", default="literature/agent_report.json")
    agent_run.add_argument("--cache-dir", default="literature/agent_cache")
    agent_run.add_argument(
        "--in-place",
        action="store_true",
        help="Allow --output to overwrite --base; a backup is created first.",
    )
    agent_run.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep rows whose five schema fields exactly match an existing row.",
    )
    build = subparsers.add_parser(
        "build", help="Merge only approved review rows into a generated CSV."
    )
    build.add_argument("--base", default="data/Electricity_complete.csv")
    build.add_argument("--review", default="literature/review_queue.csv")
    build.add_argument("--schema", default="literature/extraction_schema.json")
    build.add_argument("--output", default="data/Electricity_complete.generated.csv")
    build.add_argument(
        "--provenance", default="data/Electricity_complete.provenance.generated.csv"
    )
    build.add_argument(
        "--in-place",
        action="store_true",
        help="Allow --output to overwrite --base; a timestamped backup is created.",
    )
    return parser


def _doctor(root: Path, args: argparse.Namespace) -> int:
    checks = {
        "requests": importlib.util.find_spec("requests") is not None,
        "pymupdf": importlib.util.find_spec("pymupdf") is not None,
        "sources": _resolve(root, args.sources).exists(),
        "schema": _resolve(root, args.schema).exists(),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        print("\nInstall missing packages with: pip install -r requirements.txt", file=sys.stderr)
        return 1
    return 0


def _fetch_progress(current: int, total: int, source, status: str) -> None:
    print(f"[{current:>2}/{total}] {source.source_id}: {status}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()

    if args.command == "doctor":
        return _doctor(root, args)

    if args.command == "fetch":
        report = fetch_sources(
            project_root=root,
            sources_path=_resolve(root, args.sources),
            pdf_dir=_resolve(root, args.pdf_dir),
            report_path=_resolve(root, args.report),
            contact_email=args.email,
            limit=args.limit,
            progress=_fetch_progress,
        )
        counts: dict[str, int] = {}
        for row in report:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(json.dumps(counts, ensure_ascii=False, indent=2))
        return 0

    if args.command == "extract":
        candidates, report = extract_pdfs(
            project_root=root,
            sources_path=_resolve(root, args.sources),
            schema_path=_resolve(root, args.schema),
            pdf_dir=_resolve(root, args.pdf_dir),
            review_path=_resolve(root, args.review),
            report_path=_resolve(root, args.report),
            figure_dir=_resolve(root, args.figure_dir),
            limit=args.limit,
        )
        print(
            json.dumps(
                {
                    "sources": len(report),
                    "candidates": len(candidates),
                    "review_queue": str(_resolve(root, args.review)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run":
        pdf_dir = _resolve(root, args.pdf_dir)
        fetch_report = _resolve(root, "literature/fetch_report.csv")
        fetch_sources(
            project_root=root,
            sources_path=_resolve(root, args.sources),
            pdf_dir=pdf_dir,
            report_path=fetch_report,
            contact_email=args.email,
            limit=args.limit,
            progress=_fetch_progress,
        )
        candidates, report = extract_pdfs(
            project_root=root,
            sources_path=_resolve(root, args.sources),
            schema_path=_resolve(root, args.schema),
            pdf_dir=pdf_dir,
            review_path=_resolve(root, "literature/review_queue.csv"),
            report_path=_resolve(root, "literature/extraction_report.csv"),
            figure_dir=_resolve(root, "literature/figures"),
            limit=args.limit,
        )
        print(
            json.dumps(
                {"sources": len(report), "candidates_for_review": len(candidates)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "agent-run":
        paper_spec: str = args.papers
        paper_path = Path(paper_spec)
        if not paper_path.is_absolute() and not re.match(r"^(?:https?://|doi:)", paper_spec, re.I):
            paper_spec = str(_resolve(root, paper_spec))
        result = run_agent_pipeline(
            papers=paper_spec,
            base_path=_resolve(root, args.base),
            output_path=_resolve(root, args.output),
            provenance_path=_resolve(root, args.provenance),
            report_path=_resolve(root, args.report),
            cache_dir=_resolve(root, args.cache_dir),
            schema_path=_resolve(root, args.schema),
            in_place=args.in_place,
            deduplicate=not args.keep_duplicates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build":
        result = build_dataset(
            base_path=_resolve(root, args.base),
            review_path=_resolve(root, args.review),
            schema_path=_resolve(root, args.schema),
            output_path=_resolve(root, args.output),
            provenance_path=_resolve(root, args.provenance),
            allow_in_place=args.in_place,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
