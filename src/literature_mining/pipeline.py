"""Literature-driven carrier-mobility dataset construction utilities.

The extraction logic is material-system configurable through the supplied schema.
The repository ships with a Bi2Te3 case-study schema, while document loading,
text/table extraction, provenance capture, and dataset assembly are reusable.
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote, urljoin, urlparse


DATASET_COLUMNS = [
    "Crystal_structure",
    "temperature",
    "N_type_carrier",
    "Crystal_form",
    "Direction",
]

REVIEW_COLUMNS = [
    "candidate_id",
    "source_id",
    "doi",
    "title",
    *DATASET_COLUMNS,
    "mobility_original_value",
    "mobility_original_unit",
    "page",
    "evidence_type",
    "evidence",
    "confidence",
    "validation_notes",
    "review_status",
    "reviewer_notes",
]

FETCH_REPORT_COLUMNS = [
    "source_id",
    "doi",
    "status",
    "pdf_path",
    "pdf_url",
    "sha256",
    "license",
    "message",
]

EXTRACTION_REPORT_COLUMNS = [
    "source_id",
    "doi",
    "pdf_path",
    "pages",
    "candidate_count",
    "status",
    "review_images",
    "message",
]

SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUPERSCRIPT_TRANSLATION = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-"
)
UNICODE_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})

NUMBER_RE = re.compile(
    r"(?P<base>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\s*(?:[×x]\s*10\s*(?:\^\s*)?|[eE])\s*(?P<exp>[+\-−]?\d+))?"
)
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    doi: str = ""
    title: str = ""
    pdf_url: str = ""
    local_pdf: str = ""
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SourceRecord":
        enabled = str(row.get("enabled", "true")).strip().lower()
        return cls(
            source_id=(row.get("source_id") or row.get("key") or "").strip(),
            doi=normalize_doi(row.get("doi", "")),
            title=row.get("title", "").strip(),
            pdf_url=row.get("pdf_url", "").strip(),
            local_pdf=row.get("local_pdf", "").strip(),
            enabled=enabled not in {"0", "false", "no", "disabled"},
            notes=row.get("notes", "").strip(),
        )


@dataclass(slots=True)
class CandidateRecord:
    source_id: str
    doi: str
    title: str
    Crystal_structure: str = ""
    temperature: str = ""
    N_type_carrier: str = ""
    Crystal_form: str = ""
    Direction: str = ""
    mobility_original_value: str = ""
    mobility_original_unit: str = ""
    page: str = ""
    evidence_type: str = ""
    evidence: str = ""
    confidence: str = "0.00"
    validation_notes: str = ""
    review_status: str = "needs_review"
    reviewer_notes: str = ""
    candidate_id: str = ""

    def finalize_id(self) -> None:
        if self.candidate_id:
            return
        payload = "|".join(
            [
                self.source_id,
                self.page,
                self.Crystal_structure,
                self.temperature,
                self.N_type_carrier,
                self.evidence_type,
                self.evidence,
            ]
        )
        self.candidate_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_row(self) -> dict[str, str]:
        self.finalize_id()
        values = asdict(self)
        return {column: str(values.get(column, "")) for column in REVIEW_COLUMNS}


def normalize_doi(value: str | None) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.I)
    return text.strip().rstrip(".,;)").lower()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:120] or "source"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_sources(path: Path) -> list[SourceRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = [SourceRecord.from_row(row) for row in csv.DictReader(handle)]
    missing_ids = [index + 2 for index, record in enumerate(records) if not record.source_id]
    if missing_ids:
        raise ValueError(f"Source manifest has empty source_id/key on lines {missing_ids}")
    duplicate_ids = sorted(
        source_id
        for source_id in {record.source_id for record in records}
        if sum(item.source_id == source_id for item in records) > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate source_id values: {duplicate_ids}")
    return records


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def _is_safe_remote_url(url: str) -> tuple[bool, str]:
    """Reject local/private destinations before downloading user-configured URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "only http/https URLs are allowed"
    if parsed.username or parsed.password:
        return False, "credentials embedded in URLs are not allowed"
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        return False, "URL has no hostname"
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        return False, "local destinations are not allowed"
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        return False, f"hostname resolution failed: {exc}"
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            return False, f"non-public destination is not allowed: {ip}"
    return True, ""


def _request_session(contact_email: str = ""):
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised by doctor command
        raise RuntimeError("Missing dependency 'requests'. Run pip install -r requirements.txt") from exc
    session = requests.Session()
    contact = f"; mailto:{contact_email}" if contact_email else ""
    session.headers.update(
        {
            "User-Agent": (
                "AL-carrier-mobility-literature-pipeline/1.0 "
                "(+https://github.com/Bai-shulin/AL_for_carrier_mobility"
                f"{contact})"
            ),
            "Accept": "application/pdf, application/json;q=0.9, */*;q=0.2",
        }
    )
    return session


def resolve_open_access_locations(
    source: SourceRecord, session, contact_email: str = ""
) -> tuple[list[dict[str, str]], str]:
    """Resolve DOI metadata to openly licensed PDF locations via OpenAlex.

    OpenAlex frequently exposes a repository/PMC landing page without a
    ``pdf_url``. Those records are still useful: the NCBI OA service provides
    a lawful package containing the article PDF. Publisher PDF links are also
    supplemented with the stable ``mdpi-res`` endpoint because the public MDPI
    landing URL often returns an anti-bot HTML page to scripted clients.
    """
    locations: list[dict[str, str]] = []
    if source.pdf_url:
        locations.append({"url": source.pdf_url, "license": "user-supplied"})
    if not source.doi:
        return locations, "no DOI supplied"

    api_url = f"https://api.openalex.org/works/https://doi.org/{quote(source.doi, safe='/')}"
    params = {"mailto": contact_email} if contact_email else None
    try:
        response = session.get(
            api_url,
            params=params,
            timeout=(10, 30),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # network/API failures belong in the report
        return locations, f"OpenAlex lookup failed: {exc}"

    seen = {item["url"] for item in locations}

    def add_location(url: str, license_name: str, kind: str = "pdf") -> None:
        if url and url not in seen:
            locations.append({"url": url, "license": license_name, "kind": kind})
            seen.add(url)

    open_access = payload.get("open_access") or {}
    oa_url = open_access.get("oa_url")
    if oa_url:
        add_location(str(oa_url), "open-access")
        for fallback_url in _publisher_pdf_fallbacks(str(oa_url), source.doi):
            add_location(fallback_url, "open-access")

    for location in payload.get("locations") or []:
        is_open = bool(location.get("is_oa") or open_access.get("is_oa"))
        if not is_open:
            continue
        license_name = str(location.get("license") or "open-access")
        pdf_url = location.get("pdf_url")
        if pdf_url:
            add_location(str(pdf_url), license_name)
            for fallback_url in _publisher_pdf_fallbacks(str(pdf_url), source.doi):
                add_location(fallback_url, license_name)

        landing_url = str(location.get("landing_page_url") or "")
        pmc_match = re.search(r"(PMC\d+)|/pmc/articles/(\d+)", landing_url, flags=re.I)
        if pmc_match:
            pmc_id = pmc_match.group(1) or "PMC" + pmc_match.group(2)
            # Europe PMC serves a rendered article PDF without the NCBI
            # anti-bot interstitial and remains within the OA licence.
            add_location(
                f"https://europepmc.org/articles/{pmc_id}?pdf=render",
                license_name,
            )
            package_url = _resolve_pmc_package_url(session, pmc_id)
            if package_url:
                add_location(package_url, license_name, kind="oa_package")

    return locations, ""


def _resolve_pmc_package_url(session, pmc_id: str) -> str:
    """Return the HTTPS NCBI OA-package URL for a PMC identifier."""
    try:
        response = session.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
            params={"id": pmc_id.upper()},
            timeout=(10, 30),
            headers={"Accept": "application/xml"},
        )
        response.raise_for_status()
    except Exception:
        return ""
    match = re.search(r'<link[^>]+format="tgz"[^>]+href="([^"]+)"', response.text)
    if not match:
        return ""
    url = match.group(1).strip()
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        url = "https://ftp.ncbi.nlm.nih.gov/" + url.split("ftp.ncbi.nlm.nih.gov/", 1)[1]
    return url


def _publisher_pdf_fallbacks(url: str, doi: str = "") -> list[str]:
    """Build stable publisher PDF URLs for known OA landing URL patterns."""
    parsed = urlparse(url)
    if parsed.hostname not in {"www.mdpi.com", "mdpi.com"}:
        return []
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[-1].lower() not in {"pdf", "full"}:
        return []
    doi_tail = normalize_doi(doi).split("/", 1)[-1]
    slug_match = re.match(r"([a-z-]+?)(\d{4,})$", doi_tail, flags=re.I)
    slug = slug_match.group(1).lower() if slug_match else parts[0].lower()
    slug = {"app": "applsci"}.get(slug, slug)
    article = parts[-2].zfill(5)
    volume = parts[-4]
    if not (article.isdigit() and volume.isdigit()):
        return []
    filename = f"{slug}-{volume}-{article}"
    return [
        f"https://mdpi-res.com/d_attachment/{slug}/{filename}/article_deploy/{filename}.pdf"
    ]

def _download_pdf(session, url: str, destination: Path, max_megabytes: int = 80) -> str:
    maximum = max_megabytes * 1024 * 1024
    current_url = url
    response = None
    for _ in range(6):
        safe, reason = _is_safe_remote_url(current_url)
        if not safe:
            raise ValueError(reason)
        response = session.get(
            current_url,
            timeout=(15, 45),
            stream=True,
            allow_redirects=False,
        )
        if response.status_code not in {301, 302, 303, 307, 308}:
            break
        location = response.headers.get("location")
        response.close()
        if not location:
            raise ValueError("redirect response has no Location header")
        current_url = urljoin(current_url, location)
    else:
        raise ValueError("too many redirects")

    if response is None:  # pragma: no cover - defensive guard
        raise RuntimeError("download did not produce a response")
    with response:
        response.raise_for_status()
        advertised = int(response.headers.get("content-length", "0") or 0)
        if advertised > maximum:
            raise ValueError(f"PDF exceeds {max_megabytes} MB limit")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pdf.part")
        digest = hashlib.sha256()
        size = 0
        first_chunk = True
        try:
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 128):
                    if not chunk:
                        continue
                    if first_chunk and not chunk.lstrip().startswith(b"%PDF"):
                        raise ValueError("response is not a PDF file")
                    first_chunk = False
                    size += len(chunk)
                    if size > maximum:
                        raise ValueError(f"PDF exceeds {max_megabytes} MB limit")
                    digest.update(chunk)
                    handle.write(chunk)
            if size == 0:
                raise ValueError("empty response")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return digest.hexdigest()


def _local_pdf_path(source: SourceRecord, project_root: Path, pdf_dir: Path) -> Path:
    if source.local_pdf:
        path = Path(source.local_pdf)
        return path if path.is_absolute() else project_root / path
    return pdf_dir / f"{safe_filename(source.source_id)}.pdf"


def _download_oa_package(
    session, url: str, destination: Path, max_megabytes: int = 80
) -> str:
    """Download an NCBI OA tarball and save its largest PDF member."""
    maximum = max_megabytes * 1024 * 1024
    safe, reason = _is_safe_remote_url(url)
    if not safe:
        raise ValueError(reason)
    response = session.get(
        url,
        timeout=(15, 90),
        stream=True,
        allow_redirects=True,
    )
    with response:
        response.raise_for_status()
        advertised = int(response.headers.get("content-length", "0") or 0)
        if advertised > maximum:
            raise ValueError(f"OA package exceeds {max_megabytes} MB limit")
        payload = bytearray()
        for chunk in response.iter_content(1024 * 128):
            if chunk:
                payload.extend(chunk)
                if len(payload) > maximum:
                    raise ValueError(f"OA package exceeds {max_megabytes} MB limit")
    if not payload:
        raise ValueError("empty OA package response")

    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"invalid OA package: {exc}") from exc
    with archive:
        pdf_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".pdf")
        ]
        if not pdf_members:
            raise ValueError("OA package contains no PDF")
        member = max(pdf_members, key=lambda item: item.size)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("could not read PDF from OA package")
        pdf_bytes = extracted.read(maximum + 1)
    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise ValueError("OA package member is not a PDF")
    if len(pdf_bytes) > maximum:
        raise ValueError(f"PDF exceeds {max_megabytes} MB limit")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pdf.part")
    try:
        temporary.write_bytes(pdf_bytes)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(pdf_bytes).hexdigest()

def fetch_sources(
    project_root: Path,
    sources_path: Path,
    pdf_dir: Path,
    report_path: Path,
    contact_email: str = "",
    limit: int | None = None,
    progress: Callable[[int, int, SourceRecord, str], None] | None = None,
) -> list[dict[str, str]]:
    sources = [source for source in read_sources(sources_path) if source.enabled]
    if limit is not None:
        sources = sources[: max(0, limit)]
    session = _request_session(contact_email)
    report: list[dict[str, str]] = []

    for index, source in enumerate(sources):
        destination = _local_pdf_path(source, project_root, pdf_dir)
        row = {
            "source_id": source.source_id,
            "doi": source.doi,
            "status": "",
            "pdf_path": str(destination.relative_to(project_root))
            if destination.is_relative_to(project_root)
            else str(destination),
            "pdf_url": "",
            "sha256": "",
            "license": "",
            "message": "",
        }
        if destination.exists() and destination.read_bytes()[:4] == b"%PDF":
            row["status"] = "already_present"
            row["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            report.append(row)
            write_csv(report_path, FETCH_REPORT_COLUMNS, report)
            if progress:
                progress(index + 1, len(sources), source, row["status"])
            continue
        if source.local_pdf:
            row["status"] = "missing_local_pdf"
            row["message"] = "The configured local_pdf does not exist or is not a PDF."
            report.append(row)
            write_csv(report_path, FETCH_REPORT_COLUMNS, report)
            if progress:
                progress(index + 1, len(sources), source, row["status"])
            continue

        locations, lookup_message = resolve_open_access_locations(source, session, contact_email)
        errors: list[str] = []
        for location in locations:
            try:
                digest = (_download_oa_package if location.get("kind") == "oa_package" else _download_pdf)(session, location["url"], destination)
                row.update(
                    status="downloaded",
                    pdf_url=location["url"],
                    sha256=digest,
                    license=location.get("license", ""),
                )
                break
            except Exception as exc:
                errors.append(f"{location['url']}: {exc}")
        if not row["status"]:
            row["status"] = "needs_manual_pdf"
            messages = [message for message in [lookup_message, *errors] if message]
            row["message"] = " | ".join(messages) or "No open-access PDF location was found."
        report.append(row)
        write_csv(report_path, FETCH_REPORT_COLUMNS, report)
        if progress:
            progress(index + 1, len(sources), source, row["status"])
        if index + 1 < len(sources):
            time.sleep(0.25)

    return report


def normalize_text(text: str) -> str:
    return (
        str(text or "")
        .translate(SUBSCRIPT_TRANSLATION)
        .translate(SUPERSCRIPT_TRANSLATION)
        .translate(UNICODE_MINUS)
        .replace("µ", "μ")
        .replace("μ", "mu")
        .replace("µ", "mu")
        .replace("𝜇", "mu")
        .replace("·", " ")
    )


def parse_scientific_number(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = normalize_text(str(value)).replace(",", "").strip()
    match = NUMBER_RE.search(text)
    if not match:
        return None
    number = float(match.group("base"))
    exponent = match.group("exp")
    if exponent:
        number *= 10 ** int(exponent.translate(UNICODE_MINUS))
    return number if math.isfinite(number) else None


def _format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def _normalized_header(value: str) -> str:
    text = normalize_text(value).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _field_for_header(header: str, schema: dict[str, Any]) -> str | None:
    normalized = _normalized_header(header)
    best: tuple[int, str] | None = None
    for field, aliases in schema.get("column_aliases", {}).items():
        for alias in aliases:
            alias_normalized = _normalized_header(alias)
            if alias_normalized and alias_normalized in normalized:
                score = len(alias_normalized)
                if best is None or score > best[0]:
                    best = (score, field)
    return best[1] if best else None


def _best_table_header(rows: list[list[str]], schema: dict[str, Any]) -> tuple[int, dict[str, int], dict[str, str]]:
    if not rows:
        return 0, {}, {}
    width = max(len(row) for row in rows)
    best: tuple[int, int, dict[str, int], dict[str, str]] = (0, 0, {}, {})
    for depth in range(1, min(3, len(rows)) + 1):
        headers: list[str] = []
        for column in range(width):
            parts = [str(rows[row][column] or "") for row in range(depth) if column < len(rows[row])]
            headers.append(" ".join(part for part in parts if part.strip()))
        mapping: dict[str, int] = {}
        raw_headers: dict[str, str] = {}
        for column, header in enumerate(headers):
            field = _field_for_header(header, schema)
            if field and field not in mapping:
                mapping[field] = column
                raw_headers[field] = header
        weighted_score = len(mapping) + (2 if "N_type_carrier" in mapping else 0)
        if weighted_score > best[0]:
            best = (weighted_score, depth, mapping, raw_headers)
    return best[1], best[2], best[3]


def extract_formula(value: str, schema: dict[str, Any] | None = None) -> str:
    """Extract a formula-like token, optionally constrained by schema host elements.

    ``validation.required_elements`` is treated as a material-system filter.  The
    included Bi2Te3 schema sets it to ["Bi", "Te"], while a reusable template
    leaves it empty so the same extraction code can be applied elsewhere.
    """
    text = normalize_text(value)
    text = re.sub(r"\s+", "", text)
    text = text.replace("_{", "").replace("^{", "").replace("}", "")
    # Capture formula-like sequences, including parenthesized solid solutions.
    candidates = re.findall(
        r"(?:\([A-Z][A-Za-z0-9.+\-δxyz%]*\)[0-9.δxyz%+\-]*|"
        r"[A-Z][a-z]?[0-9.δxyz%+\-]*){2,}",
        text,
    )
    required_elements = set(
        (schema or {}).get("validation", {}).get("required_elements", [])
    )
    candidates = [
        candidate.strip(".,;:")
        for candidate in candidates
        if not required_elements
        or required_elements.issubset(set(ELEMENT_RE.findall(candidate)))
    ]
    return max(candidates, key=len) if candidates else ""


def _mobility_multiplier(header_or_unit: str) -> tuple[float, str, str]:
    text = _normalized_header(header_or_unit)
    compact = re.sub(r"\s+", "", text)
    exponent_match = re.search(r"10(?:\^)?([+\-]?\d+)", compact)
    header_scale = 10 ** int(exponent_match.group(1)) if exponent_match else 1.0
    if "cm2" in compact or "cm^2" in compact:
        return header_scale, "cm2 V-1 s-1", ""
    if "m2" in compact or "m^2" in compact:
        return header_scale * 1e4, "m2 V-1 s-1", ""
    return 1.0, "", "mobility unit not recognized; value was not rescaled"


def _parse_temperature(value: str, header: str = "") -> tuple[float | None, str]:
    combined = _normalized_header(f"{header} {value}")
    if "room temperature" in combined or combined.strip() in {"rt", "r.t."}:
        return 300.0, "room temperature normalized to 300 K"
    number = parse_scientific_number(value)
    if number is None:
        return None, "temperature missing"
    if "°c" in combined or "deg c" in combined or "celsius" in combined:
        return number + 273.15, "temperature converted from Celsius to kelvin"
    return number, ""


def _infer_code(context: str, mapping_config: dict[str, Any]) -> tuple[str, str]:
    if not mapping_config.get("enabled", False):
        return "", "categorical mapping disabled pending scientific confirmation"
    normalized = _normalized_header(context)
    matches: set[str] = set()
    for phrase, code in mapping_config.get("values", {}).items():
        if _normalized_header(phrase) in normalized:
            matches.add(str(code))
    if len(matches) == 1:
        return matches.pop(), ""
    if len(matches) > 1:
        return "", "conflicting categorical descriptions"
    default = mapping_config.get("default", "")
    return str(default), "categorical value defaulted" if default != "" else "categorical value missing"


def validate_candidate(candidate: CandidateRecord, schema: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    formula = candidate.Crystal_structure
    elements = set(ELEMENT_RE.findall(formula))
    required_elements = set(schema.get("validation", {}).get("required_elements", []))
    if not formula:
        notes.append("chemical formula missing")
    elif required_elements and not required_elements.issubset(elements):
        notes.append(f"formula does not contain required elements {sorted(required_elements)}")
    if re.search(r"[δxyz]", formula):
        notes.append("formula contains a symbolic stoichiometry variable")

    temperature = parse_scientific_number(candidate.temperature)
    temperature_range = schema.get("validation", {}).get("temperature_K", [0, 2000])
    if temperature is None:
        notes.append("temperature missing")
    elif not temperature_range[0] <= temperature <= temperature_range[1]:
        notes.append("temperature outside configured range")

    mobility = parse_scientific_number(candidate.N_type_carrier)
    mobility_range = schema.get("validation", {}).get("mobility_cm2_Vs", [0, 10000])
    if mobility is None:
        notes.append("mobility missing")
    elif not mobility_range[0] < mobility <= mobility_range[1]:
        notes.append("mobility outside configured range")

    for field in ("Crystal_form", "Direction"):
        allowed = {str(item) for item in schema.get("validation", {}).get(field, [])}
        value = str(getattr(candidate, field))
        if value == "":
            notes.append(f"{field} missing")
        elif allowed and value not in allowed:
            notes.append(f"{field} has invalid code {value!r}")
    return notes


def _table_cell(row: Sequence[Any], column: int | None) -> str:
    if column is None or column >= len(row):
        return ""
    return str(row[column] or "").strip()


def extract_table_candidates(
    rows: list[list[str]],
    source: SourceRecord,
    schema: dict[str, Any],
    page_number: int,
    page_context: str = "",
) -> list[CandidateRecord]:
    depth, mapping, headers = _best_table_header(rows, schema)
    if "N_type_carrier" not in mapping:
        return []

    candidates: list[CandidateRecord] = []
    mobility_header = headers.get("N_type_carrier", "")
    multiplier, canonical_unit, unit_note = _mobility_multiplier(mobility_header)
    crystal_code, crystal_note = _infer_code(
        page_context, schema.get("categorical_mappings", {}).get("Crystal_form", {})
    )
    direction_code, direction_note = _infer_code(
        page_context, schema.get("categorical_mappings", {}).get("Direction", {})
    )

    for row in rows[depth:]:
        mobility_raw = _table_cell(row, mapping.get("N_type_carrier"))
        mobility = parse_scientific_number(mobility_raw)
        if mobility is None:
            continue
        mobility *= multiplier

        formula_cell = _table_cell(row, mapping.get("Crystal_structure"))
        formula = extract_formula(formula_cell, schema) or extract_formula(" ".join(map(str, row)), schema)
        temperature_raw = _table_cell(row, mapping.get("temperature"))
        temperature, temperature_note = _parse_temperature(
            temperature_raw, headers.get("temperature", "")
        )
        row_context = " | ".join(str(cell or "").strip() for cell in row)

        confidence = 0.45
        confidence += 0.20 if formula else 0
        confidence += 0.12 if temperature is not None else 0
        confidence += 0.10 if canonical_unit else 0
        confidence += 0.04 if crystal_code else 0
        confidence += 0.04 if direction_code else 0
        notes = [note for note in [unit_note, temperature_note, crystal_note, direction_note] if note]
        candidate = CandidateRecord(
            source_id=source.source_id,
            doi=source.doi,
            title=source.title,
            Crystal_structure=formula,
            temperature=_format_number(temperature),
            N_type_carrier=_format_number(mobility),
            Crystal_form=crystal_code,
            Direction=direction_code,
            mobility_original_value=mobility_raw,
            mobility_original_unit=mobility_header,
            page=str(page_number),
            evidence_type="table",
            evidence=row_context[:1200],
            confidence=f"{min(confidence, 0.95):.2f}",
        )
        notes.extend(validate_candidate(candidate, schema))
        candidate.validation_notes = "; ".join(dict.fromkeys(notes))
        candidate.finalize_id()
        candidates.append(candidate)
    return candidates


def _text_table_candidates(
    page_text: str,
    source: SourceRecord,
    schema: dict[str, Any],
    page_number: int,
) -> list[CandidateRecord]:
    """Recover simple vertically extracted tables when MuPDF finds no grid.

    Many publisher PDFs expose each table cell on its own text line.  The
    mobility column is still unambiguous when a header contains a mobility
    unit: a formula row is followed by numeric cells in header order.  This
    fallback intentionally emits review candidates only; it does not infer
    values from plotted curves.
    """
    lines = [
        re.sub(r"\s+", " ", normalize_text(line)).strip()
        for line in page_text.splitlines()
        if line.strip()
    ]
    header_indices = [
        index
        for index, line in enumerate(lines)
        if re.search(r"mobility|\bmu\s*[_-]?\s*[hH]?\b", line, re.I)
        and _mobility_multiplier(line)[1]
    ]
    if not header_indices:
        return []

    header_index = header_indices[0]
    mobility_header = lines[header_index]
    multiplier, canonical_unit, unit_note = _mobility_multiplier(mobility_header)
    context = " ".join(lines[max(0, header_index - 12) : header_index + 1])
    temperature_match = re.search(
        r"(?:at|near|temperature(?:\s+of)?)\s*"
        r"(room temperature|RT|[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:K|deg\s*C|C))",
        context,
        re.I,
    )
    temperature, temperature_note = _parse_temperature(
        temperature_match.group(1) if temperature_match else ""
    )
    crystal_code, crystal_note = _infer_code(
        page_text, schema.get("categorical_mappings", {}).get("Crystal_form", {})
    )
    direction_code, direction_note = _infer_code(
        page_text, schema.get("categorical_mappings", {}).get("Direction", {})
    )

    numeric_pattern = re.compile(
        r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:\s*[xX]\s*10\s*\^?\s*[+\-]?\s*\d+|[eE][+\-]?\d+)?$"
    )
    candidates: list[CandidateRecord] = []
    for row_index in range(header_index + 1, len(lines)):
        formula = extract_formula(lines[row_index], schema)
        if not formula:
            continue
        numeric_values: list[tuple[str, float]] = []
        for value_index in range(row_index + 1, min(len(lines), row_index + 9)):
            value_text = normalize_text(lines[value_index])
            if extract_formula(lines[value_index], schema):
                break
            if not numeric_pattern.fullmatch(value_text):
                if numeric_values:
                    break
                continue
            value = parse_scientific_number(value_text)
            if value is None:
                break
            numeric_values.append((lines[value_index], value))
        # A Hall-mobility table normally has conductivity, concentration,
        # mobility, Seebeck coefficient, and power factor in that order.
        if len(numeric_values) < 3:
            continue
        raw_value, value = numeric_values[2]
        mobility = value * multiplier
        candidate = CandidateRecord(
            source_id=source.source_id,
            doi=source.doi,
            title=source.title,
            Crystal_structure=formula,
            temperature=_format_number(temperature),
            N_type_carrier=_format_number(mobility),
            Crystal_form=crystal_code,
            Direction=direction_code,
            mobility_original_value=raw_value,
            mobility_original_unit=mobility_header,
            page=str(page_number),
            evidence_type="table_text",
            evidence=" | ".join(lines[max(header_index, row_index - 1) : row_index + 1 + len(numeric_values)])[:1200],
            confidence="0.82" if temperature is not None else "0.70",
        )
        notes = [
            "table recovered from vertically extracted PDF text",
            unit_note,
            temperature_note,
            crystal_note,
            direction_note,
        ]
        notes.extend(validate_candidate(candidate, schema))
        candidate.validation_notes = "; ".join(dict.fromkeys(note for note in notes if note))
        candidate.finalize_id()
        candidates.append(candidate)
    return candidates

def _mobility_from_text(text: str) -> tuple[float | None, str, str]:
    normalized = normalize_text(text)
    mobility_word = r"(?:hall\s+|carrier\s+|electron\s+)?mobility|\bmu\s*[_-]?\s*[hH]?\b"
    unit = r"(?:cm\s*\^?\s*2|cm2|m\s*\^?\s*2|m2)\s*(?:/|\s)*(?:V|v)"
    patterns = [
        re.compile(rf"(?:{mobility_word}).{{0,80}}?(?P<number>{NUMBER_RE.pattern}).{{0,35}}?(?P<unit>{unit}[^,;.\n]*)", re.I),
        re.compile(rf"(?P<number>{NUMBER_RE.pattern}).{{0,25}}?(?P<unit>{unit}[^,;.\n]*).{{0,80}}?(?:{mobility_word})", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        raw = match.group("number")
        value = parse_scientific_number(raw)
        multiplier, canonical, _ = _mobility_multiplier(match.group("unit"))
        if value is not None:
            return value * multiplier, raw, match.group("unit") if canonical else ""
    return None, "", ""


def _parallel_list_candidates(
    page_text: str,
    source: SourceRecord,
    schema: dict[str, Any],
    page_number: int,
) -> list[CandidateRecord]:
    """Extract lists explicitly paired with sample lists by 'respectively'."""
    text = re.sub(r"\s+", " ", normalize_text(page_text))
    number_token = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*[×x]\s*10\s*\^?\s*[+\-]?\d+)?"
    pattern = re.compile(
        rf"(?:carrier\s+concentration\s+and\s+mobility|"
        rf"mobility\s+and\s+carrier\s+concentration).{{0,280}}?"
        rf"\([^)]*cm\s*[-]?\s*3[^)]*\)\s*and\s*"
        rf"(?P<values>{number_token}(?:\s*,\s*{number_token})+)\s*"
        rf"\((?P<unit>[^)]*(?:cm\s*2|m\s*2)[^)]*)\)+\s*for\s*"
        rf"(?P<formulas>.{{1,600}}?)\s*,?\s*respectively",
        re.I,
    )
    candidates: list[CandidateRecord] = []
    for match in pattern.finditer(text):
        raw_values = [item.strip() for item in match.group("values").split(",")]
        values = [parse_scientific_number(item) for item in raw_values]
        formulas = [
            extract_formula(item, schema)
            for item in re.split(r"\s*,\s*", match.group("formulas"))
        ]
        formulas = [formula for formula in formulas if formula]
        if not values or any(value is None for value in values) or len(values) != len(formulas):
            continue
        multiplier, canonical_unit, unit_note = _mobility_multiplier(match.group("unit"))
        crystal_code, crystal_note = _infer_code(
            text, schema.get("categorical_mappings", {}).get("Crystal_form", {})
        )
        direction_code, direction_note = _infer_code(
            text, schema.get("categorical_mappings", {}).get("Direction", {})
        )
        evidence = match.group(0)[:1200]
        for formula, raw_value, value in zip(formulas, raw_values, values):
            candidate = CandidateRecord(
                source_id=source.source_id,
                doi=source.doi,
                title=source.title,
                Crystal_structure=formula,
                temperature="",
                N_type_carrier=_format_number(float(value) * multiplier),
                Crystal_form=crystal_code,
                Direction=direction_code,
                mobility_original_value=raw_value,
                mobility_original_unit=match.group("unit"),
                page=str(page_number),
                evidence_type="parallel_text_list",
                evidence=evidence,
                confidence="0.78" if canonical_unit else "0.66",
            )
            notes = [
                note
                for note in [unit_note, crystal_note, direction_note]
                if note
            ]
            notes.extend(validate_candidate(candidate, schema))
            candidate.validation_notes = "; ".join(dict.fromkeys(notes))
            candidate.finalize_id()
            candidates.append(candidate)
    return candidates


def extract_text_candidates(
    page_text: str,
    source: SourceRecord,
    schema: dict[str, Any],
    page_number: int,
) -> list[CandidateRecord]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n|(?<=[.;])\s+", page_text) if block.strip()]
    candidates = _parallel_list_candidates(page_text, source, schema, page_number)
    candidates.extend(_text_table_candidates(page_text, source, schema, page_number))
    for block in blocks:
        if not re.search(r"mobility|\bmu\s*[_-]?\s*[hH]?\b", normalize_text(block), re.I):
            continue
        mobility, raw_value, raw_unit = _mobility_from_text(block)
        if mobility is None:
            continue
        formula = extract_formula(block, schema)
        temperature_match = re.search(
            r"(?:at|near|temperature(?:\s+of)?)\s*(room temperature|RT|\d+(?:\.\d+)?\s*(?:K|°C))",
            normalize_text(block),
            re.I,
        )
        temperature, temperature_note = _parse_temperature(
            temperature_match.group(1) if temperature_match else ""
        )
        crystal_code, crystal_note = _infer_code(
            block, schema.get("categorical_mappings", {}).get("Crystal_form", {})
        )
        direction_code, direction_note = _infer_code(
            block, schema.get("categorical_mappings", {}).get("Direction", {})
        )
        candidate = CandidateRecord(
            source_id=source.source_id,
            doi=source.doi,
            title=source.title,
            Crystal_structure=formula,
            temperature=_format_number(temperature),
            N_type_carrier=_format_number(mobility),
            Crystal_form=crystal_code,
            Direction=direction_code,
            mobility_original_value=raw_value,
            mobility_original_unit=raw_unit,
            page=str(page_number),
            evidence_type="text",
            evidence=re.sub(r"\s+", " ", block)[:1200],
            confidence="0.68" if formula and temperature is not None else "0.52",
        )
        notes = [note for note in [temperature_note, crystal_note, direction_note] if note]
        notes.extend(validate_candidate(candidate, schema))
        candidate.validation_notes = "; ".join(dict.fromkeys(notes))
        candidate.finalize_id()
        candidates.append(candidate)
    return candidates


def _deduplicate_candidates(candidates: Iterable[CandidateRecord]) -> list[CandidateRecord]:
    best_by_key: dict[tuple[str, ...], CandidateRecord] = {}
    for candidate in candidates:
        key = (
            candidate.source_id,
            re.sub(r"\s+", "", candidate.Crystal_structure),
            candidate.temperature,
            candidate.N_type_carrier,
            candidate.page,
        )
        current = best_by_key.get(key)
        if current is None or float(candidate.confidence) > float(current.confidence):
            best_by_key[key] = candidate
    return sorted(
        best_by_key.values(),
        key=lambda item: (item.source_id, int(item.page or 0), item.candidate_id),
    )


def _write_review_queue_preserving_decisions(
    path: Path, candidates: Iterable[CandidateRecord]
) -> None:
    """Refresh extraction output without erasing reviewed/manual records."""
    new_rows = [candidate.as_row() for candidate in candidates]
    old_rows = _read_csv_rows(path) if path.exists() else []
    old_by_id = {
        row.get("candidate_id", ""): row
        for row in old_rows
        if row.get("candidate_id", "")
    }
    editable_columns = [
        *DATASET_COLUMNS,
        "review_status",
        "reviewer_notes",
    ]
    refreshed_ids: set[str] = set()
    merged: list[dict[str, str]] = []
    for row in new_rows:
        candidate_id = row["candidate_id"]
        refreshed_ids.add(candidate_id)
        previous = old_by_id.get(candidate_id)
        if previous and (
            previous.get("review_status", "").strip().lower()
            in {"approved", "rejected"}
            or previous.get("reviewer_notes", "").strip()
        ):
            for column in editable_columns:
                row[column] = previous.get(column, row.get(column, ""))
        merged.append(row)

    # Keep manually entered rows and completed decisions even if a later PDF
    # extraction no longer rediscovers the exact candidate.
    for row in old_rows:
        if row.get("candidate_id", "") in refreshed_ids:
            continue
        status = row.get("review_status", "").strip().lower()
        if row.get("evidence_type", "").strip().lower() == "manual" or status in {
            "approved",
            "rejected",
        }:
            merged.append({column: row.get(column, "") for column in REVIEW_COLUMNS})
    write_csv(path, REVIEW_COLUMNS, merged)


def extract_pdfs(
    project_root: Path,
    sources_path: Path,
    schema_path: Path,
    pdf_dir: Path,
    review_path: Path,
    report_path: Path,
    figure_dir: Path | None = None,
    limit: int | None = None,
) -> tuple[list[CandidateRecord], list[dict[str, str]]]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pymupdf'. Run pip install -r requirements.txt") from exc

    schema = load_json(schema_path)
    sources = [source for source in read_sources(sources_path) if source.enabled]
    if limit is not None:
        sources = sources[: max(0, limit)]

    all_candidates: list[CandidateRecord] = []
    report: list[dict[str, str]] = []
    for source in sources:
        pdf_path = _local_pdf_path(source, project_root, pdf_dir)
        row = {
            "source_id": source.source_id,
            "doi": source.doi,
            "pdf_path": str(pdf_path),
            "pages": "0",
            "candidate_count": "0",
            "status": "",
            "review_images": "",
            "message": "",
        }
        if not pdf_path.exists():
            row.update(status="missing_pdf", message="Run fetch or provide local_pdf in sources.csv.")
            report.append(row)
            continue

        source_candidates: list[CandidateRecord] = []
        mobility_mentions = 0
        review_images: list[str] = []
        try:
            document = pymupdf.open(pdf_path)
            row["pages"] = str(len(document))
            for page_number, page in enumerate(document, start=1):
                # Keep the PDF's native block order. Sorting by coordinates can
                # interleave two-column articles and break value/sample pairing.
                page_text = page.get_text("text", sort=False)
                page_mentions = len(
                    re.findall(r"mobility|\bmu\s*[_-]?\s*[hH]?\b", page_text, re.I)
                )
                mobility_mentions += page_mentions
                before_page = len(source_candidates)
                source_candidates.extend(extract_text_candidates(page_text, source, schema, page_number))
                try:
                    table_finder = page.find_tables()
                    tables = table_finder.tables
                except Exception:
                    tables = []
                for table in tables:
                    extracted = table.extract()
                    table_rows = [
                        [str(cell or "") for cell in row_values]
                        for row_values in extracted
                        if row_values
                    ]
                    source_candidates.extend(
                        extract_table_candidates(
                            table_rows, source, schema, page_number, page_text
                        )
                    )
                if (
                    figure_dir is not None
                    and page_mentions
                    and len(source_candidates) == before_page
                ):
                    figure_dir.mkdir(parents=True, exist_ok=True)
                    image_path = figure_dir / (
                        f"{safe_filename(source.source_id)}_page_{page_number}.png"
                    )
                    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(image_path)
                    try:
                        review_images.append(str(image_path.relative_to(project_root)))
                    except ValueError:
                        review_images.append(str(image_path))
            document.close()
            source_candidates = _deduplicate_candidates(source_candidates)
            all_candidates.extend(source_candidates)
            row["candidate_count"] = str(len(source_candidates))
            row["review_images"] = " | ".join(review_images)
            if source_candidates:
                row["status"] = "candidates_extracted"
                row["message"] = "Review every candidate before approval."
            elif mobility_mentions:
                row["status"] = "manual_digitization_required"
                row["message"] = (
                    "Mobility is discussed but no machine-readable value was found; "
                    "inspect the exported pages, plots, captions, or supplementary data."
                )
            else:
                row["status"] = "no_target_mention"
                row["message"] = "No mobility mention was found in extracted PDF text."
        except Exception as exc:
            row.update(status="extraction_failed", message=str(exc))
        report.append(row)

    deduplicated = _deduplicate_candidates(all_candidates)
    _write_review_queue_preserving_decisions(review_path, deduplicated)
    write_csv(report_path, EXTRACTION_REPORT_COLUMNS, report)
    return deduplicated, report


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _dataset_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    formula = re.sub(r"\s+", "", row.get("Crystal_structure", ""))
    temperature = parse_scientific_number(row.get("temperature", ""))
    mobility = parse_scientific_number(row.get("N_type_carrier", ""))
    return (
        formula,
        f"{temperature:.8g}" if temperature is not None else "",
        f"{mobility:.10g}" if mobility is not None else "",
        str(row.get("Crystal_form", "")).strip(),
        str(row.get("Direction", "")).strip(),
    )


def _validate_approved_row(row: dict[str, str], schema: dict[str, Any]) -> list[str]:
    candidate = CandidateRecord(
        source_id=row.get("source_id", ""),
        doi=row.get("doi", ""),
        title=row.get("title", ""),
        Crystal_structure=re.sub(r"\s+", "", row.get("Crystal_structure", "")),
        temperature=row.get("temperature", ""),
        N_type_carrier=row.get("N_type_carrier", ""),
        Crystal_form=row.get("Crystal_form", ""),
        Direction=row.get("Direction", ""),
    )
    return validate_candidate(candidate, schema)


def build_dataset(
    base_path: Path,
    review_path: Path,
    schema_path: Path,
    output_path: Path,
    provenance_path: Path,
    allow_in_place: bool = False,
) -> dict[str, int]:
    """Merge approved review rows while preserving the five-column model schema."""
    if output_path.resolve() == base_path.resolve() and not allow_in_place:
        raise ValueError("Refusing to overwrite the base dataset without allow_in_place=True")

    schema = load_json(schema_path)
    base_rows = _read_csv_rows(base_path)
    review_rows = _read_csv_rows(review_path)
    approved = [
        row for row in review_rows if row.get("review_status", "").strip().lower() == "approved"
    ]

    errors: list[str] = []
    for row in approved:
        row_errors = _validate_approved_row(row, schema)
        if row_errors:
            identifier = row.get("candidate_id") or row.get("source_id") or "unknown"
            errors.append(f"{identifier}: {'; '.join(row_errors)}")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Approved rows failed validation:\n{preview}")

    merged = [{column: row.get(column, "") for column in DATASET_COLUMNS} for row in base_rows]
    existing_keys = {_dataset_key(row) for row in merged}
    provenance: list[dict[str, str]] = []
    added = 0
    duplicates = 0

    for row in approved:
        dataset_row = {column: str(row.get(column, "")).strip() for column in DATASET_COLUMNS}
        dataset_row["Crystal_structure"] = re.sub(r"\s+", "", dataset_row["Crystal_structure"])
        key = _dataset_key(dataset_row)
        if key in existing_keys:
            duplicates += 1
            continue
        merged.append(dataset_row)
        existing_keys.add(key)
        added += 1
        provenance.append(
            {
                **{column: dataset_row[column] for column in DATASET_COLUMNS},
                "candidate_id": row.get("candidate_id", ""),
                "source_id": row.get("source_id", ""),
                "doi": normalize_doi(row.get("doi", "")),
                "page": row.get("page", ""),
                "evidence_type": row.get("evidence_type", ""),
                "evidence": row.get("evidence", ""),
                "reviewer_notes": row.get("reviewer_notes", ""),
                "merged_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    if output_path.resolve() == base_path.resolve() and base_path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = base_path.with_name(f"{base_path.stem}.backup-{stamp}{base_path.suffix}")
        shutil.copy2(base_path, backup)

    write_csv(output_path, DATASET_COLUMNS, merged)
    provenance_columns = [
        *DATASET_COLUMNS,
        "candidate_id",
        "source_id",
        "doi",
        "page",
        "evidence_type",
        "evidence",
        "reviewer_notes",
        "merged_at_utc",
    ]
    write_csv(provenance_path, provenance_columns, provenance)
    return {
        "base_rows": len(base_rows),
        "approved_rows": len(approved),
        "added_rows": added,
        "duplicate_rows": duplicates,
        "output_rows": len(merged),
    }
