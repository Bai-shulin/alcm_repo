"""Input-driven literature extraction Agent for carrier-mobility datasets.

The Agent processes only documents explicitly supplied by the caller. Material-
system filters and extraction aliases are supplied through a schema, allowing the
same document/extraction workflow to support the included Bi2Te3 case study or
other material families.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence
from urllib.parse import urljoin

from .pipeline import (
    DATASET_COLUMNS,
    CandidateRecord,
    SourceRecord,
    _deduplicate_candidates,
    _is_safe_remote_url,
    extract_table_candidates,
    extract_text_candidates,
    normalize_doi,
    normalize_text,
    write_csv,
)


@dataclass(slots=True)
class PaperInput:
    """One user-supplied document reference."""

    value: str
    source_id: str = ""
    doi: str = ""
    title: str = ""
    notes: str = ""

    @classmethod
    def from_value(cls, value: str | dict[str, Any] | "PaperInput") -> "PaperInput":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            reference = (
                value.get("path")
                or value.get("file")
                or value.get("pdf")
                or value.get("url")
                or value.get("doi")
                or value.get("source")
                or ""
            )
            doi = normalize_doi(str(value.get("doi", "")))
            if str(reference).lower().startswith("doi:"):
                doi = normalize_doi(str(reference)[4:])
            return cls(
                value=str(reference).strip(),
                source_id=str(value.get("source_id") or value.get("id") or "").strip(),
                doi=doi,
                title=str(value.get("title") or "").strip(),
                notes=str(value.get("notes") or "").strip(),
            )
        text = str(value).strip()
        doi = normalize_doi(text[4:]) if text.lower().startswith("doi:") else ""
        return cls(value=text, doi=doi)

    @property
    def stable_id(self) -> str:
        if self.source_id:
            return self.source_id
        basis = self.doi or self.value
        return "paper-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class PaperDocument:
    """Normalized document representation consumed by the extraction Agent."""

    paper: PaperInput
    source: SourceRecord
    kind: str
    page_texts: list[str]
    tables: list[tuple[str, list[list[str]]]] = field(default_factory=list)
    local_path: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(self.page_texts)


@dataclass(slots=True)
class PaperDecision:
    relevant: bool
    status: str
    reason: str
    candidate_count: int


@dataclass(slots=True)
class PaperResult:
    paper: PaperInput
    decision: PaperDecision
    rows: list[dict[str, str]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


class PaperJudge(Protocol):
    """Replaceable judgment layer; no vendor-specific LLM is required."""

    def assess(
        self, document: PaperDocument, candidates: Sequence[CandidateRecord]
    ) -> PaperDecision:
        ...


class HeuristicPaperJudge:
    """Deterministic baseline judge for mobility relevance and data presence."""

    _target_terms = re.compile(
        r"\b(?:hall\s+)?mobility\b|\b(?:electron|carrier)\s+mobility\b|"
        r"\bcarrier\s+concentration\b|\bmu\s*[_-]?\s*h\b",
        re.I,
    )

    def assess(
        self, document: PaperDocument, candidates: Sequence[CandidateRecord]
    ) -> PaperDecision:
        text = normalize_text(document.text)
        mentions = self._target_terms.findall(text)
        if not mentions:
            return PaperDecision(False, "not_relevant", "no mobility or carrier-transport target mention", 0)
        if not candidates:
            return PaperDecision(
                False,
                "mention_without_extractable_values",
                "mobility is discussed but no numeric sample-level value was extracted",
                0,
            )
        return PaperDecision(True, "candidates_extracted", "sample-level mobility candidates found", len(candidates))


class _HTMLParser(HTMLParser):
    """Small dependency-free HTML text/table parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            self._row.append(re.sub(r"\s+", " ", " ".join(self._cell_parts)).strip())
            self._cell_parts = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            if any(cell for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        self.text_parts.append(data)


class DocumentLoader:
    """Load only user-supplied files/URLs/DOIs into a normalized document."""

    def __init__(self, cache_dir: Path = Path("literature/agent_cache"), session: Any = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.session = session
        if self.session is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("requests is required for URL/DOI inputs") from exc
            self.session = requests.Session()
            self.session.headers.update(
                {
                    "User-Agent": "AL-carrier-mobility-input-agent/1.0",
                    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.1",
                }
            )

    def load(self, paper: PaperInput) -> PaperDocument:
        value = paper.value
        local = Path(value)
        if local.exists() and local.is_file():
            payload = local.read_bytes()
            return self._from_bytes(paper, payload, local.suffix.lower(), str(local))

        url = value
        if value.lower().startswith("doi:"):
            doi = normalize_doi(value[4:])
            paper.doi = paper.doi or doi
            url = "https://doi.org/" + doi
        if not re.match(r"https?://", url, re.I):
            raise FileNotFoundError(f"paper input is not a local file, URL, or DOI: {value}")
        content, content_type, final_url = self._fetch(url)
        suffix = ".pdf" if content.startswith(b"%PDF") or "pdf" in content_type.lower() else ".html"
        if suffix == ".html":
            pdf_url = self._find_pdf_link(content.decode("utf-8", errors="replace"), final_url)
            if pdf_url:
                try:
                    pdf_content, pdf_type, pdf_final = self._fetch(pdf_url)
                    if pdf_content.startswith(b"%PDF") or "pdf" in pdf_type.lower():
                        return self._from_bytes(paper, pdf_content, ".pdf", pdf_final)
                except Exception:
                    pass
        return self._from_bytes(paper, content, suffix, final_url)

    def _fetch(self, url: str) -> tuple[bytes, str, str]:
        safe, reason = _is_safe_remote_url(url)
        if not safe:
            raise ValueError(reason)
        response = self.session.get(url, timeout=(15, 90), allow_redirects=True)
        response.raise_for_status()
        content = response.content
        if len(content) > 80 * 1024 * 1024:
            raise ValueError("document exceeds 80 MB limit")
        return content, str(response.headers.get("content-type", "")), str(response.url)

    @staticmethod
    def _find_pdf_link(markup: str, base_url: str) -> str:
        patterns = [
            r"<meta[^>]+name=[\"']citation_pdf_url[\"'][^>]+content=[\"']([^\"']+)",
            r"<link[^>]+type=[\"']application/pdf[\"'][^>]+href=[\"']([^\"']+)",
            r"<a[^>]+href=[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']",
        ]
        for pattern in patterns:
            match = re.search(pattern, markup, re.I)
            if match:
                return urljoin(base_url, match.group(1))
        return ""

    def _from_bytes(
        self, paper: PaperInput, payload: bytes, suffix: str, origin: str
    ) -> PaperDocument:
        source = SourceRecord(paper.stable_id, normalize_doi(paper.doi), paper.title)
        if suffix.lower() == ".pdf" or payload.startswith(b"%PDF"):
            try:
                import pymupdf
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pymupdf is required for PDF inputs") from exc
            document = pymupdf.open(stream=payload, filetype="pdf")
            metadata = document.metadata or {}
            if not paper.title:
                source.title = str(metadata.get("title") or "").strip()
            pages: list[str] = []
            tables: list[tuple[str, list[list[str]]]] = []
            for page_number, page in enumerate(document, 1):
                page_text = page.get_text("text", sort=False)
                pages.append(page_text)
                try:
                    found = page.find_tables().tables
                except Exception:
                    found = []
                for table_number, table in enumerate(found, 1):
                    rows = [[str(cell or "") for cell in row] for row in table.extract() if row]
                    if rows:
                        tables.append((f"{page_number}:{table_number}", rows))
            document.close()
            local_path = self._cache_bytes(paper.stable_id, payload, ".pdf")
            return PaperDocument(paper, source, "pdf", pages, tables, str(local_path))

        parser = _HTMLParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
        title = paper.title
        if not title:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", payload.decode("utf-8", errors="replace"), re.I | re.S)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
            source.title = title
        tables = [("html:%d" % index, rows) for index, rows in enumerate(parser.tables, 1)]
        return PaperDocument(paper, source, "html", [" ".join(parser.text_parts)], tables, origin)

    def _cache_bytes(self, source_id: str, payload: bytes, suffix: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (re.sub(r"[^A-Za-z0-9._-]", "_", source_id) + suffix)
        if not path.exists() or path.read_bytes() != payload:
            path.write_bytes(payload)
        return path


def load_paper_inputs(spec: Any) -> list[PaperInput]:
    """Load a list, CSV/JSON manifest, or one explicit paper reference."""
    if isinstance(spec, (list, tuple)):
        return [PaperInput.from_value(item) for item in spec]
    if isinstance(spec, PaperInput):
        return [spec]
    path = Path(str(spec))
    if path.exists() and path.is_file():
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [PaperInput.from_value(row) for row in csv.DictReader(handle)]
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("papers", data.get("inputs", []))
            return [PaperInput.from_value(item) for item in data]
        if path.suffix.lower() in {".txt", ".list"}:
            return [PaperInput.from_value(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [PaperInput.from_value(str(spec))]


def _candidate_row(candidate: CandidateRecord) -> dict[str, str]:
    return {column: str(getattr(candidate, column, "") or "") for column in DATASET_COLUMNS}


def _candidate_provenance(
    document: PaperDocument,
    candidate: CandidateRecord,
    table_number: str = "",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_id": document.source.source_id,
        "doi": document.source.doi,
        "title": document.source.title,
        "source_input": document.paper.value,
        "document_kind": document.kind,
        "page": candidate.page,
        "table_number": table_number,
        "figure_number": "",
        "source_text": candidate.evidence,
        "extraction_method": candidate.evidence_type,
        "confidence": candidate.confidence,
        "validation_notes": candidate.validation_notes,
        "dataset_row": _candidate_row(candidate),
    }


class LiteratureExtractionAgent:
    """Orchestrate loading, judgment, extraction, and schema mapping."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        judge: PaperJudge | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.judge = judge or HeuristicPaperJudge()
        self.schema = schema

    def process(self, paper: PaperInput) -> PaperResult:
        try:
            document = self.loader.load(paper)
            candidates: list[CandidateRecord] = []
            table_numbers: dict[str, str] = {}
            for page_number, page_text in enumerate(document.page_texts, 1):
                page_candidates = extract_text_candidates(
                    page_text, document.source, self._schema, page_number
                )
                candidates.extend(page_candidates)
                for candidate in page_candidates:
                    table_numbers[candidate.candidate_id] = ""
            for table_number, rows in document.tables:
                table_prefix = table_number.split(":", 1)[0]
                page_number = int(table_prefix) if table_prefix.isdigit() else 1
                page_text = document.page_texts[page_number - 1] if document.kind == "pdf" and page_number <= len(document.page_texts) else document.text
                table_candidates = extract_table_candidates(
                    rows, document.source, self._schema, page_number, page_text
                )
                candidates.extend(table_candidates)
                for candidate in table_candidates:
                    table_numbers[candidate.candidate_id] = table_number
            candidates = _deduplicate_candidates(candidates)
            decision = self.judge.assess(document, candidates)
            if not decision.relevant:
                return PaperResult(paper, decision)
            provenance = [
                _candidate_provenance(document, candidate, table_numbers.get(candidate.candidate_id, ""))
                for candidate in candidates
            ]
            return PaperResult(paper, decision, [_candidate_row(candidate) for candidate in candidates], provenance)
        except Exception as exc:
            decision = PaperDecision(False, "error", str(exc), 0)
            return PaperResult(paper, decision, error=str(exc))

    @property
    def _schema(self) -> dict[str, Any]:
        if self.schema is not None:
            return self.schema
        return {
            "column_aliases": {
                "Crystal_structure": ["composition", "compound", "material", "sample", "chemical formula"],
                "temperature": ["temperature", "temp"],
                "N_type_carrier": ["mobility", "hall mobility", "electron mobility", "carrier mobility", "mu"],
                "Crystal_form": ["crystal form", "sample form", "morphology"],
                "Direction": ["direction", "orientation", "measurement direction"],
            },
            "categorical_mappings": {
                "Crystal_form": {
                    "enabled": True,
                    "values": {
                        "polycrystalline": 0,
                        "polycrystal": 0,
                        "single crystal": 1,
                        "single-crystalline": 1,
                        "thin film": 1,
                    },
                },
                "Direction": {
                    "enabled": True,
                    "values": {
                        "unspecified": 0,
                        "parallel to pressing direction": 1,
                        "perpendicular to pressing direction": 2,
                        "in-plane": 1,
                        "cross-plane": 2,
                    },
                },
            },
            "validation": {"required_elements": []},
        }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", "", str(row.get(column, ""))) for column in DATASET_COLUMNS)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_agent_pipeline(
    papers: Any,
    base_path: Path = Path("data/Electricity_complete.csv"),
    output_path: Path = Path("data/Electricity_complete.agent.csv"),
    provenance_path: Path = Path("data/Electricity_complete.agent.provenance.jsonl"),
    report_path: Path = Path("literature/agent_report.json"),
    cache_dir: Path = Path("literature/agent_cache"),
    schema_path: Path | None = None,
    in_place: bool = False,
    deduplicate: bool = True,
    agent: LiteratureExtractionAgent | None = None,
) -> dict[str, Any]:
    """Run the complete input-driven pipeline and write schema-compatible output."""
    if output_path.resolve() == base_path.resolve() and not in_place:
        raise ValueError("Refusing to overwrite the base dataset without in_place=True")
    inputs = load_paper_inputs(papers)
    schema = None
    if schema_path is not None:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    extractor = agent or LiteratureExtractionAgent(
        loader=DocumentLoader(cache_dir), schema=schema
    )
    base_rows = _read_rows(base_path)
    merged = [{column: row.get(column, "") for column in DATASET_COLUMNS} for row in base_rows]
    existing = {_row_key(row) for row in merged}
    report: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    extracted_rows = 0
    added_rows = 0
    duplicates = 0

    for paper in inputs:
        result = extractor.process(paper)
        report.append(
            {
                "source_id": paper.stable_id,
                "source": paper.value,
                "doi": paper.doi,
                "status": result.decision.status,
                "reason": result.decision.reason,
                "candidate_count": len(result.rows),
                "error": result.error,
            }
        )
        extracted_rows += len(result.rows)
        for row, provenance_row in zip(result.rows, result.provenance):
            key = _row_key(row)
            is_duplicate = deduplicate and key in existing
            provenance_row = dict(provenance_row)
            provenance_row["merged"] = not is_duplicate
            provenance.append(provenance_row)
            if is_duplicate:
                duplicates += 1
                continue
            merged.append(row)
            existing.add(key)
            added_rows += 1

    if output_path.resolve() == base_path.resolve() and base_path.exists():
        backup = base_path.with_name(base_path.stem + ".backup-agent" + base_path.suffix)
        shutil.copy2(base_path, backup)
    write_csv(output_path, DATASET_COLUMNS, merged)
    _write_jsonl(provenance_path, provenance)
    report_payload = {
        "input_count": len(inputs),
        "base_rows": len(base_rows),
        "papers_with_candidates": sum(item["candidate_count"] > 0 for item in report),
        "extracted_rows": extracted_rows,
        "added_rows": added_rows,
        "duplicate_rows": duplicates,
        "output_rows": len(merged),
        "papers": report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload