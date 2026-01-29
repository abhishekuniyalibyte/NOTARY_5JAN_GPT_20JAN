import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import difflib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine
from src.phase3_document_intake import DocumentIntake, DocumentTypeDetector, FileFormat
from src.phase4_text_extraction import (
    CollectionExtractionResult,
    DataExtractor,
    DocumentExtractionResult,
    ExtractedData,
    TextExtractor,
    TextNormalizer,
)
from src.phase5_legal_validation import LegalValidator
from src.phase6_gap_detection import ActionPriority, Gap, GapDetector, GapType
from src.phase7_data_update import DataUpdater
from src.phase8_final_confirmation import FinalConfirmationEngine
from src.phase9_certificate_generation import CertificateGenerator
from src.phase10_notary_review import NotaryReviewSystem, ReviewStatus
from src.phase11_final_output import FinalOutputGenerator


DEFAULT_SUMMARY_PATH = "cetificate from dataset/certificate_summary.json"
DEFAULT_CATALOG_PATH = "cetificate from dataset/client_file_catalogs.json"
DEFAULT_CERT_TYPE = "certificacion_de_firmas"
DEFAULT_PURPOSE = "para_bps"
DEFAULT_EXTRACTION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
DEFAULT_ANALYSIS_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_QA_MODEL = DEFAULT_ANALYSIS_MODEL
MAX_LLM_CHARS = 3000
MAX_QA_CONTEXT_CHARS = 8000
MAX_QA_DOC_CHARS = 4000


def get_default_option(options: List[Dict[str, str]], value: str) -> Dict[str, str]:
    for option in options:
        if option["value"] == value:
            return option
    return options[0] if options else {"value": value, "label": value}


def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_customer_key(value: str) -> str:
    value_norm = normalize_text(value)
    if not value_norm:
        return ""
    suffixes = {"sa", "srl", "ltda", "sas"}
    tokens = [token for token in value_norm.split() if token not in suffixes]
    return " ".join(tokens).strip()


def infer_catalog_customer(subject_name: str, catalog_customers: List[str]) -> Optional[str]:
    if not subject_name or not catalog_customers:
        return None
    target = normalize_customer_key(subject_name)
    for customer in catalog_customers:
        if normalize_customer_key(customer) == target:
            return customer
    return None


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "status" not in data:
        data["status"] = "ok"
    return data


@st.cache_data(show_spinner=False)
def load_client_catalog(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    with open(catalog_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_text_for_llm(file_path: str) -> str:
    try:
        document = DocumentIntake.process_file(file_path)
        text, _, _ = extract_text_without_ocr(document)
        return text
    except Exception:
        return ""


def truncate_text(value: str, limit: int) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    truncated = value[:limit]
    if " " in truncated:
        trimmed = truncated.rsplit(" ", 1)[0]
        if trimmed:
            truncated = trimmed
    return f"{truncated}..."


def coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "null"):
            return None
        return cleaned
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in ("none", "null"):
        return None
    return cleaned


def ensure_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned = str(item).strip()
            if not cleaned or cleaned.lower() in ("none", "null"):
                continue
            cleaned_items.append(cleaned)
        return cleaned_items
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "null"):
            return []
        return [cleaned]
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in ("none", "null"):
        return []
    return [cleaned]


def format_confidence(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def build_document_type_detail(
    selected_type: str,
    type_source: str,
    detected_type: Optional[str],
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
) -> str:
    llm_type = llm_result.get("certificate_type") if llm_result else None
    llm_conf = llm_result.get("confidence") if llm_result else None
    llm_status = llm_result.get("status") if llm_result else None
    keyword_type = keyword_result.get("certificate_type") if keyword_result else None
    keyword_conf = keyword_result.get("confidence") if keyword_result else None
    keyword_status = keyword_result.get("status") if keyword_result else None

    return (
        f"selected={selected_type}; source={type_source}; "
        f"detected={detected_type or 'n/a'}; "
        f"llm={llm_type or 'n/a'} ({format_confidence(llm_conf)}, {llm_status or 'n/a'}); "
        f"keyword={keyword_type or 'n/a'} ({format_confidence(keyword_conf)}, {keyword_status or 'n/a'})"
    )


def build_detected_type_detail(
    detected_type: Optional[str],
    catalog_info: Optional[Dict[str, Any]],
) -> str:
    parts = []
    if detected_type:
        parts.append(detected_type)
    if catalog_info:
        description = catalog_info.get("description")
        if description:
            parts.append(description)
    return " - ".join(parts)


def collect_error_reasons(
    validation_status: str,
    extraction_success: Optional[bool],
    extraction_error: Optional[str],
    llm_extraction_error: Optional[str],
    text_extraction_error: Optional[str],
    ocr_error: Optional[str],
    processing_status: Optional[str],
) -> List[str]:
    reasons = []
    if processing_status == "error":
        reasons.append("processing_status=error")
    if extraction_success is False:
        reasons.append("extraction_success=false")
    if extraction_error:
        reasons.append(f"extraction_error: {extraction_error}")
    if llm_extraction_error:
        reasons.append(f"llm_extraction_error: {llm_extraction_error}")
    if text_extraction_error:
        reasons.append(f"text_extraction_error: {text_extraction_error}")
    if ocr_error:
        reasons.append(f"ocr_error: {ocr_error}")
    if validation_status == "invalid":
        reasons.append("validation_status=invalid")
    return reasons


def format_has_error_flag(has_error: Optional[bool]) -> str:
    return "yes" if has_error else "no"


def format_has_error_detail(has_error: Optional[bool], error_reasons: List[str]) -> str:
    if not has_error:
        return "no"
    if error_reasons:
        return f"yes: {' | '.join(error_reasons)}"
    return "yes"


def is_zip_file(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def extract_text_from_doc_via_libreoffice(file_path: Path) -> Tuple[str, Optional[str]]:
    soffice_path = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_path:
        return "", "LibreOffice is required to extract legacy .doc files."

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            soffice_path,
            "--headless",
            "--convert-to",
            "txt:Text",
            "--outdir",
            tmpdir,
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            return "", f"LibreOffice conversion failed: {stderr or 'unknown error'}"

        output_path = Path(tmpdir) / f"{file_path.stem}.txt"
        if not output_path.exists():
            txt_files = list(Path(tmpdir).glob("*.txt"))
            if not txt_files:
                return "", "LibreOffice conversion succeeded but no output file found."
            output_path = txt_files[0]

        try:
            return output_path.read_text(encoding="utf-8", errors="ignore"), None
        except OSError as exc:
            return "", f"Failed to read converted text: {exc}"


def extract_text_without_ocr(document) -> Tuple[str, str, Optional[str]]:
    file_path = Path(document.file_path)
    file_format = getattr(document, "file_format", None)
    format_value = file_format.value if file_format else file_path.suffix.lower().lstrip(".")

    if format_value == "txt":
        try:
            return TextExtractor.extract_from_text_file(file_path), "text", None
        except Exception as exc:
            return "", "text", f"Text extraction failed: {exc}"
    if format_value == "pdf":
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            return "", "text", f"PyPDF2 is required for PDF extraction: {exc}"
        try:
            reader = PdfReader(str(file_path))
            text_chunks = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(text_chunks).strip(), "text", None
        except Exception as exc:
            return "", "text", f"PDF extraction failed: {exc}"
    if format_value == "docx":
        try:
            return TextExtractor.extract_from_docx(file_path), "text", None
        except Exception as exc:
            return "", "text", f"DOCX extraction failed: {exc}"
    if format_value == "doc":
        if is_zip_file(file_path):
            try:
                return TextExtractor.extract_from_docx(file_path), "text", None
            except Exception as exc:
                return "", "text", f"DOCX extraction failed: {exc}"

        text, error = extract_text_from_doc_via_libreoffice(file_path)
        if text:
            return text, "text", None
        if error:
            return "", "text", error

        try:
            import textract
        except Exception as exc:
            return "", "text", (
                "Legacy .doc extraction requires textract or a conversion to .docx "
                f"(import failed: {exc})"
            )
        try:
            text_bytes = textract.process(str(file_path))
            return text_bytes.decode("utf-8", errors="ignore"), "text", None
        except Exception as exc:
            return "", "text", f"Legacy .doc extraction failed: {exc}"

    return "", "none", f"Unsupported file format: {format_value or 'unknown'}"


def call_groq_extraction(
    model: str,
    api_key: str,
    doc_text: str,
    filename: str,
) -> Dict[str, Any]:
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}
    if not doc_text.strip():
        return {"status": "error", "message": "No text provided for LLM extraction."}

    client = Groq(api_key=api_key)
    prompt = (
        "You extract structured data from notarial documents.\n"
        "Reply with JSON only (no Markdown).\n"
        "Keys:\n"
        "company_name, rut, ci, registro_comercio, acta_number, padron_bps, dates, emails.\n"
        "Use null when missing and [] for lists.\n\n"
        f"Filename: {filename}\n\n"
        "Document text:\n"
        f"{doc_text[:MAX_LLM_CHARS]}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_from_text(content)
        if parsed is not None:
            return parsed
        return {
            "status": "error",
            "message": "LLM did not return valid JSON.",
            "raw": content,
        }
    except Exception as exc:
        return {"status": "error", "message": f"Groq request failed: {exc}"}


def apply_llm_fields(extracted_data: ExtractedData, llm_payload: Dict[str, Any]) -> None:
    extracted_data.company_name = coerce_optional_str(llm_payload.get("company_name")) or extracted_data.company_name
    extracted_data.rut = coerce_optional_str(llm_payload.get("rut")) or extracted_data.rut
    extracted_data.ci = coerce_optional_str(llm_payload.get("ci")) or extracted_data.ci
    extracted_data.registro_comercio = (
        coerce_optional_str(llm_payload.get("registro_comercio"))
        or extracted_data.registro_comercio
    )
    extracted_data.acta_number = coerce_optional_str(llm_payload.get("acta_number")) or extracted_data.acta_number
    extracted_data.padron_bps = coerce_optional_str(llm_payload.get("padron_bps")) or extracted_data.padron_bps

    dates = ensure_list(llm_payload.get("dates"))
    if dates:
        extracted_data.dates = dates
    emails = ensure_list(llm_payload.get("emails"))
    if emails:
        extracted_data.emails = emails


def apply_regex_fallback(extracted_data: ExtractedData, normalized_text: str) -> None:
    if not normalized_text:
        return
    if not extracted_data.company_name:
        extracted_data.company_name = DataExtractor.extract_company_name(normalized_text)
    if not extracted_data.rut:
        extracted_data.rut = DataExtractor.extract_rut(normalized_text)
    if not extracted_data.ci:
        extracted_data.ci = DataExtractor.extract_ci(normalized_text)
    if not extracted_data.registro_comercio:
        extracted_data.registro_comercio = DataExtractor.extract_registro_comercio(normalized_text)
    if not extracted_data.acta_number:
        extracted_data.acta_number = DataExtractor.extract_acta_number(normalized_text)
    if not extracted_data.padron_bps:
        extracted_data.padron_bps = DataExtractor.extract_padron_bps(normalized_text)
    if not extracted_data.dates:
        extracted_data.dates = DataExtractor.extract_dates(normalized_text)
    if not extracted_data.emails:
        extracted_data.emails = DataExtractor.extract_emails(normalized_text)


def process_collection_with_llm(
    collection,
    llm_settings: Dict[str, str],
) -> CollectionExtractionResult:
    result = CollectionExtractionResult(collection=collection)

    for document in collection.documents:
        raw_text = ""
        base_method = "none"
        base_error = None
        ocr_used = False
        ocr_error = None

        try:
            raw_text, base_method, base_error = extract_text_without_ocr(document)
        except Exception as exc:
            base_error = str(exc)

        if not raw_text and llm_settings.get("ocr_fallback"):
            file_format = getattr(document, "file_format", None)
            if file_format in (FileFormat.PDF, FileFormat.JPG, FileFormat.JPEG, FileFormat.PNG):
                try:
                    raw_text, extraction_method = TextExtractor.extract_text(document)
                    base_method = extraction_method
                    ocr_used = extraction_method == "ocr"
                except Exception as exc:
                    ocr_error = str(exc)

        normalized_text = TextNormalizer.normalize_text(raw_text or "")

        # If filename-based detection failed, try content-based detection so users can name files anything.
        if not document.detected_type and normalized_text:
            inferred_type = DocumentTypeDetector.detect_from_text(normalized_text)
            if inferred_type:
                document.detected_type = inferred_type
                if isinstance(document.metadata, dict):
                    document.metadata["detected_type_source"] = "content"

        extraction_method = base_method if base_method in ("text", "ocr") else "none"
        extracted_data = ExtractedData(
            document_type=document.detected_type,
            raw_text=raw_text or "",
            normalized_text=normalized_text,
            extraction_method=extraction_method,
        )

        if not raw_text:
            extracted_data.additional_fields["extraction_warning"] = (
                "No text extracted. Enable OCR fallback to read scanned documents."
            )
            extracted_data.confidence = 0.2
        else:
            extracted_data.confidence = 0.7 if llm_settings.get("enabled") else 1.0
        if base_error:
            extracted_data.additional_fields["text_extraction_error"] = base_error
        if ocr_error:
            extracted_data.additional_fields["ocr_error"] = ocr_error
        if ocr_used:
            extracted_data.additional_fields["ocr_used"] = True

        llm_payload = None
        if llm_settings.get("enabled") and raw_text:
            llm_payload = call_groq_extraction(
                model=llm_settings.get("extraction_model", DEFAULT_EXTRACTION_MODEL),
                api_key=llm_settings.get("api_key", ""),
                doc_text=raw_text,
                filename=document.file_name,
            )
            if llm_payload.get("status") == "error":
                extracted_data.additional_fields["llm_extraction_error"] = llm_payload.get("message")
            else:
                apply_llm_fields(extracted_data, llm_payload)
                extracted_data.additional_fields["llm_extraction"] = llm_payload

        apply_regex_fallback(extracted_data, normalized_text)

        result.extraction_results.append(
            DocumentExtractionResult(
                document=document,
                extracted_data=extracted_data,
                success=True,
            )
        )

    return result


def normalize_purpose(value: str) -> str:
    if not value:
        return ""
    value = value.lower()
    value = value.replace("para_", "").replace("_", " ")
    return normalize_text(value)


def make_filename_keys(filename: str) -> List[str]:
    if not filename:
        return []
    path = Path(filename)
    keys = {
        normalize_text(path.name),
        normalize_text(path.stem),
    }
    return [key for key in keys if key]


def detect_pan_card_hint(doc_text: str) -> Optional[str]:
    text_norm = normalize_text(doc_text)
    if not text_norm:
        return None
    if (
        "pan card" in text_norm
        or "pancard" in text_norm
        or "permanent account number" in text_norm
    ):
        return "pan_card"
    return None


@st.cache_data(show_spinner=False)
def build_llm_reference(summary: Dict[str, Any]) -> Dict[str, Any]:
    reference = {}
    for cert_type, info in summary.get("identified_certificate_types", {}).items():
        reference[cert_type] = {
            "count": info.get("count", 0),
            "purposes": list(info.get("purposes", {}).keys()),
            "attributes": info.get("attributes", []),
            "examples": info.get("examples", [])[:3],
        }
    return reference


def keyword_classification(doc_text: str, summary_reference: Dict[str, Any]) -> Dict[str, Any]:
    text_norm = normalize_text(doc_text)
    if not text_norm:
        return {"status": "error", "message": "No text for keyword classification."}

    best_type = None
    best_score = 0
    matched_attrs: List[str] = []

    for cert_type, info in summary_reference.items():
        attrs = [normalize_text(attr) for attr in info.get("attributes", [])]
        score = 0
        hits = []
        for attr in attrs:
            if attr and attr in text_norm:
                score += 1
                hits.append(attr)
        if score > best_score:
            best_score = score
            best_type = cert_type
            matched_attrs = hits

    detected_purpose = ""
    for cert_type, info in summary_reference.items():
        for purpose in info.get("purposes", []):
            purpose_norm = normalize_purpose(purpose)
            if purpose_norm and purpose_norm in text_norm:
                detected_purpose = purpose
                break
        if detected_purpose:
            break

    if best_type and best_score > 0:
        confidence = min(0.8, 0.2 + 0.1 * best_score)
        return {
            "status": "ok",
            "is_certificate": True,
            "certificate_type": best_type,
            "purpose": detected_purpose or "",
            "confidence": confidence,
            "reason": f"Matched attributes: {', '.join(matched_attrs)}" if matched_attrs else "Matched attributes.",
        }

    return {"status": "error", "message": "No keyword match found."}


def map_summary_type_to_intent(cert_type: str) -> str:
    cert_norm = normalize_text(cert_type)
    if "firma" in cert_norm:
        return "certificacion_de_firmas"
    if "personeria" in cert_norm:
        return "certificado_de_personeria"
    if "representacion" in cert_norm:
        return "certificado_de_representacion"
    if "vigencia" in cert_norm:
        return "certificado_de_vigencia"
    if "situacion" in cert_norm or "juridica" in cert_norm:
        return "certificado_de_situacion_juridica"
    if "poder" in cert_norm:
        return "poder_general"
    return "otros"


def map_summary_purpose_to_intent(purpose: str) -> str:
    if not purpose:
        return "otros"
    purpose_norm = normalize_text(purpose)
    if purpose.lower().startswith("para_"):
        return purpose.lower()

    mapping = {
        "bps": "para_bps",
        "abitab": "para_abitab",
        "dgi": "para_dgi",
        "zona franca": "para_zona_franca",
        "zona_franca": "para_zona_franca",
        "zonafranca": "para_zona_franca",
        "msp": "para_msp",
        "ute": "para_ute",
        "antel": "para_antel",
        "rupe": "para_rupe",
        "mef": "para_mef",
        "imm": "para_imm",
        "mtop": "para_mtop",
        "migraciones": "para_migraciones",
        "banco": "para_banco",
        "compraventa": "para_compraventa",
        "base de datos": "para_base_datos",
        "base datos": "para_base_datos",
    }

    for key, value in mapping.items():
        if key in purpose_norm:
            return value
    return "otros"


def is_positive_classification(result: Optional[Dict[str, Any]]) -> bool:
    if not result or result.get("status") == "error":
        return False
    if result.get("is_certificate") is False:
        return False
    cert_type = result.get("certificate_type")
    if not cert_type or cert_type == "non_certificate":
        return False
    return True


def choose_classification(
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if is_positive_classification(llm_result):
        return llm_result
    if is_positive_classification(keyword_result):
        return keyword_result
    return None


def is_result_ok(result: Optional[Dict[str, Any]]) -> bool:
    return bool(result) and result.get("status") != "error"


def is_keyword_ok(result: Optional[Dict[str, Any]]) -> bool:
    return bool(result) and result.get("status") == "ok"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def detect_classification_conflict(
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not is_result_ok(llm_result) or not is_keyword_ok(keyword_result):
        return None
    llm_is_cert = llm_result.get("is_certificate")
    keyword_is_cert = keyword_result.get("is_certificate")
    if llm_is_cert is False and keyword_is_cert is True:
        return "Conflicto: LLM indica no certificado, keywords indican certificado."
    if llm_is_cert is True and keyword_is_cert is True:
        llm_type = llm_result.get("certificate_type")
        keyword_type = keyword_result.get("certificate_type")
        if llm_type and keyword_type and llm_type != keyword_type:
            return "Conflicto: tipos de certificado difieren entre LLM y keywords."
    return None


def build_review_reasons(
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
    match_result: Optional[Dict[str, Any]],
) -> List[str]:
    reasons: List[str] = []

    conflict = detect_classification_conflict(llm_result, keyword_result)
    if conflict:
        reasons.append(conflict)

    if not is_result_ok(llm_result) and not is_keyword_ok(keyword_result):
        reasons.append("No se pudo clasificar por contenido.")

    if match_result:
        status = match_result.get("status")
        if status == "needs_review":
            reasons.append(match_result.get("reason") or "Requiere verificación manual.")
        elif status == "not_found":
            reasons.append("No se encontró coincidencia en certificate_summary.json.")

    return dedupe_preserve_order(reasons)


def build_review_queue(per_file_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    queue = []
    for item in per_file_data.values():
        if not item.get("review_required"):
            continue
        queue.append(
            {
                "filename": item.get("filename", "documento"),
                "reasons": item.get("review_reasons", []) or [],
            }
        )
    return queue


def build_review_gaps(review_queue: List[Dict[str, Any]]) -> List[Gap]:
    gaps = []
    for entry in review_queue:
        filename = entry.get("filename", "documento")
        reasons = entry.get("reasons", []) or []
        description = "; ".join(reasons) if reasons else "Documento requiere verificación manual."
        gaps.append(
            Gap(
                gap_type=GapType.REVIEW_REQUIRED,
                priority=ActionPriority.URGENT,
                title=f"Revisar documento: {filename}",
                description=description,
                current_state="No verificado",
                required_state="Verificado por notario o fuente oficial",
                action_required="Verificar manualmente el documento",
            )
        )
    return gaps


def derive_intent_override(classification: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not is_positive_classification(classification):
        return None
    cert_type = map_summary_type_to_intent(classification.get("certificate_type", ""))
    if cert_type == "otros":
        return None
    purpose = map_summary_purpose_to_intent(classification.get("purpose", ""))
    return {"certificate_type": cert_type, "purpose": purpose}


def choose_intent_override(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    overrides = []
    for candidate in candidates:
        classification = candidate.get("classification")
        if not classification:
            continue
        override = derive_intent_override(classification)
        if override:
            overrides.append((override, candidate))

    if not overrides:
        return None

    cert_types = {item[0]["certificate_type"] for item in overrides}
    if len(cert_types) != 1:
        return None

    best_override, best_candidate = max(
        overrides,
        key=lambda item: float(item[1].get("confidence", 0.0)),
    )
    result = dict(best_override)
    result["source"] = best_candidate.get("source") or "unknown"
    result["confidence"] = float(best_candidate.get("confidence", 0.0))
    result["filename"] = best_candidate.get("filename")
    return result


def call_groq_classification(
    model: str,
    api_key: str,
    doc_text: str,
    summary_reference: Dict[str, Any],
) -> Dict[str, Any]:
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}

    client = Groq(api_key=api_key)
    context_lines = []
    for cert_type, info in summary_reference.items():
        purposes = ", ".join(info.get("purposes", [])) or "none"
        examples = "; ".join(info.get("examples", [])) or "none"
        context_lines.append(
            f"- {cert_type}: purposes={purposes}; examples={examples}"
        )
    context_text = "\n".join(context_lines)

    prompt = (
        "You classify Uruguayan notarial documents.\n"
        "Use ONLY the categories provided. Reply with JSON only.\n"
        "Do not use Markdown or code fences.\n\n"
        "Categories (from certificate_summary.json):\n"
        f"{context_text}\n\n"
        "Return JSON with keys:\n"
        "is_certificate (true/false), certificate_type, purpose, confidence (0-1), reason.\n"
        "If non-certificate, set certificate_type='non_certificate'.\n\n"
        "Document text:\n"
        f"{doc_text[:MAX_LLM_CHARS]}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise document classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_from_text(content)
        if parsed is not None:
            return parsed
        return {
            "status": "error",
            "message": "LLM did not return valid JSON.",
            "raw": content,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Groq request failed: {exc}",
        }


def build_file_context(file_result: Dict[str, Any], include_text: bool = True) -> str:
    filename = file_result.get("filename", "document")
    document_type = file_result.get("document_type") or "unknown"
    validation = file_result.get("validation", {})
    match_result = file_result.get("match") or {}
    doc_text = ""
    if include_text:
        doc_text = truncate_text(file_result.get("doc_text", ""), MAX_QA_DOC_CHARS)

    lines = [
        f"Filename: {filename}",
        f"Document type: {document_type}",
        f"Validation: {validation.get('status')} - {validation.get('reason')}",
    ]
    if match_result:
        lines.append(
            f"Dataset match: {match_result.get('status')} - {match_result.get('reason')}"
        )
    if doc_text:
        lines.append("Content:")
        lines.append(doc_text)
    return "\n".join(lines).strip()


def build_qa_context(
    file_results: List[Dict[str, Any]],
    certificate_text: str,
    scope_key: str,
) -> str:
    if scope_key == "certificate":
        if not certificate_text:
            return ""
        content = truncate_text(certificate_text, MAX_QA_CONTEXT_CHARS)
        return f"Generated certificate text:\n{content}"
    if scope_key.startswith("file:"):
        try:
            index = int(scope_key.split(":", 1)[1])
        except ValueError:
            return ""
        if 0 <= index < len(file_results):
            content = build_file_context(file_results[index], include_text=True)
            return truncate_text(content, MAX_QA_CONTEXT_CHARS)
        return ""

    total = len(file_results)
    type_counts: Dict[str, int] = {}
    doc_lines = []
    for file_result in file_results:
        doc_type = file_result.get("document_type") or "unknown"
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        filename = file_result.get("filename", "document")
        validation = file_result.get("validation", {})
        doc_lines.append(
            f"- {filename} | type={doc_type} | validation={validation.get('status')}"
        )

    type_summary = ", ".join(f"{doc_type} ({count})" for doc_type, count in type_counts.items())
    header_lines = [
        f"Total documents: {total}",
        f"Document types: {type_summary or 'unknown'}",
        "Documents:",
    ]
    combined = "\n".join(header_lines + doc_lines)
    return truncate_text(combined, MAX_QA_CONTEXT_CHARS)


def call_groq_document_qa(
    model: str,
    api_key: str,
    question: str,
    context: str,
) -> Dict[str, Any]:
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}
    if not question or not question.strip():
        return {"status": "error", "message": "Question is empty."}
    if not context or not context.strip():
        return {"status": "error", "message": "No document context available for Q&A."}

    client = Groq(api_key=api_key)
    prompt = (
        "You answer questions about processed notarial documents.\n"
        "Use ONLY the context below. If the answer is not in the context, say so.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer briefly and clearly."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful document Q&A assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return {"status": "ok", "answer": content.strip()}
    except Exception as exc:
        return {"status": "error", "message": f"Groq request failed: {exc}"}


@st.cache_data(show_spinner=False)
def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def build_summary_index(summary: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []

    for group, group_entries in summary.get("certificate_file_mapping", {}).items():
        for entry in group_entries:
            item = dict(entry)
            item["group"] = group
            item["entry_type"] = "certificate"
            entries.append(item)

    for entry in summary.get("non_certificate_documents", []):
        item = dict(entry)
        item["group"] = "non_certificate"
        item["entry_type"] = "non_certificate"
        entries.append(item)

    filename_index: Dict[str, List[Dict[str, Any]]] = {}
    customer_index: Dict[str, List[Dict[str, Any]]] = {}
    all_filenames_display: List[str] = []
    all_customers_display: List[str] = []

    for entry in entries:
        filename = entry.get("filename") or entry.get("path") or ""
        if filename:
            all_filenames_display.append(filename)
        for key in make_filename_keys(filename):
            filename_index.setdefault(key, []).append(entry)

        customer = entry.get("customer") or ""
        if customer:
            all_customers_display.append(customer)
        customer_key = normalize_text(customer)
        if customer_key:
            customer_index.setdefault(customer_key, []).append(entry)

    return {
        "entries": entries,
        "filename_index": filename_index,
        "customer_index": customer_index,
        "all_filenames_display": sorted(set(all_filenames_display)),
        "all_customers_display": sorted(set(all_customers_display)),
    }


def is_certificate_entry(entry: Dict[str, Any]) -> bool:
    return entry.get("entry_type") == "certificate"


def entry_has_error(entry: Dict[str, Any]) -> bool:
    return bool(entry.get("error_flag"))


def purpose_matches(entry_purpose: str, user_purpose: str) -> bool:
    if not entry_purpose or not user_purpose:
        return False
    entry_norm = normalize_purpose(entry_purpose)
    user_norm = normalize_purpose(user_purpose)
    if not entry_norm or not user_norm:
        return False
    return entry_norm == user_norm or entry_norm in user_norm or user_norm in entry_norm


def top_fuzzy_matches(query: str, candidates: List[str], limit: int = 5) -> List[Tuple[str, float]]:
    query_norm = normalize_text(query)
    if not query_norm:
        return []
    scored = []
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        ratio = difflib.SequenceMatcher(None, query_norm, candidate_norm).ratio()
        scored.append((candidate, ratio))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(cand, score) for cand, score in scored[:limit] if score > 0]


def dedupe_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for entry in entries:
        key = (entry.get("customer"), entry.get("filename"), entry.get("path"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def match_document(
    filename: str,
    subject_name: str,
    extracted_company: Optional[str],
    purpose_value: str,
    summary_index: Dict[str, Any],
    llm_result: Optional[Dict[str, Any]] = None,
    keyword_result: Optional[Dict[str, Any]] = None,
    content_text: str = "",
    content_only: bool = False,
) -> Dict[str, Any]:
    llm_ok = is_result_ok(llm_result)
    keyword_ok = is_keyword_ok(keyword_result)
    llm_non_certificate = llm_ok and llm_result.get("is_certificate") is False
    keyword_certificate = keyword_ok and keyword_result.get("is_certificate") is True
    if llm_non_certificate and keyword_certificate:
        return {
            "status": "needs_review",
            "match_type": "llm_keyword_conflict",
            "confidence": float(llm_result.get("confidence", 0.0)) if llm_ok else 0.0,
            "reason": "Conflicto entre LLM (no certificado) y keywords (certificado).",
            "matches": [],
            "llm_result": llm_result,
            "keyword_result": keyword_result,
        }
    if llm_non_certificate and not keyword_certificate:
        return {
            "status": "not_applicable",
            "match_type": "non_certificate",
            "confidence": float(llm_result.get("confidence", 0.0)) if llm_ok else 0.0,
            "reason": "Documento clasificado como no certificado; match del dataset omitido.",
            "matches": [],
            "llm_result": llm_result,
            "keyword_result": keyword_result,
        }

    if not content_only:
        filename_keys = make_filename_keys(filename)
        matched_entries: List[Dict[str, Any]] = []

        for key in filename_keys:
            matched_entries.extend(summary_index["filename_index"].get(key, []))

        matched_entries = dedupe_entries(matched_entries)

        if matched_entries:
            cert_entries = [e for e in matched_entries if is_certificate_entry(e)]
            non_cert_entries = [e for e in matched_entries if not is_certificate_entry(e)]
            if cert_entries:
                if any(entry_has_error(entry) for entry in cert_entries):
                    return {
                        "status": "not_found",
                        "match_type": "filename_error",
                        "confidence": 0.6,
                        "reason": "Filename matched, but entry is flagged as error in dataset.",
                        "matches": cert_entries,
                    }
                return {
                    "status": "correct",
                    "match_type": "filename",
                    "confidence": 1.0,
                    "reason": "Exact filename match found in certificate dataset.",
                    "matches": cert_entries,
                }
            if non_cert_entries:
                if llm_result and llm_result.get("is_certificate") is True:
                    return {
                        "status": "needs_review",
                        "match_type": "filename_non_certificate_llm_conflict",
                        "confidence": 0.6,
                        "reason": "Filename is non-certificate, but LLM classified as certificate.",
                        "matches": non_cert_entries,
                        "llm_result": llm_result,
                    }
                return {
                    "status": "not_found",
                    "match_type": "filename_non_certificate",
                    "confidence": 0.8,
                    "reason": "Filename matched, but document is classified as non-certificate.",
                    "matches": non_cert_entries,
                }

    customer_keys = []
    if subject_name:
        customer_keys.append(normalize_text(subject_name))
    if extracted_company and normalize_text(extracted_company) not in customer_keys:
        customer_keys.append(normalize_text(extracted_company))

    customer_matches: List[Dict[str, Any]] = []
    for key in customer_keys:
        customer_matches.extend(summary_index["customer_index"].get(key, []))
    customer_matches = dedupe_entries(customer_matches)

    if customer_matches:
        purpose_matches_entries = [
            entry for entry in customer_matches
            if purpose_matches(entry.get("purpose", ""), purpose_value)
        ]
        cert_entries = [e for e in purpose_matches_entries if is_certificate_entry(e)]
        if cert_entries:
            return {
                "status": "correct",
                "match_type": "customer_purpose",
                "confidence": 0.7,
                "reason": "Customer and purpose match found in certificate dataset.",
                "matches": cert_entries,
            }
        return {
            "status": "not_found",
            "match_type": "customer_only",
            "confidence": 0.5,
            "reason": "Customer match found, but no purpose match for this document.",
            "matches": customer_matches[:5],
        }

    if llm_result and llm_result.get("is_certificate") is True:
        llm_type = llm_result.get("certificate_type", "")
        llm_purpose = llm_result.get("purpose", "")
        summary_types = summary_index.get("summary_reference", {})
        if llm_type in summary_types:
            purpose_list = summary_types[llm_type].get("purposes", [])
            if not purpose_list or normalize_purpose(llm_purpose) in [
                normalize_purpose(p) for p in purpose_list
            ]:
                return {
                    "status": "correct",
                    "match_type": "llm_only",
                    "confidence": float(llm_result.get("confidence", 0.5)),
                    "reason": "LLM classified document type matches summary taxonomy.",
                    "matches": [],
                    "llm_result": llm_result,
                }

    if content_text:
        if not keyword_result:
            keyword_result = keyword_classification(content_text, summary_index.get("summary_reference", {}))
        if keyword_result.get("status") == "ok":
            return {
                "status": "correct",
                "match_type": "content_keywords",
                "confidence": float(keyword_result.get("confidence", 0.5)),
                "reason": keyword_result.get("reason", "Keyword match from content."),
                "matches": [],
                "llm_result": llm_result,
                "keyword_result": keyword_result,
            }

    filename_suggestions = top_fuzzy_matches(filename, summary_index["all_filenames_display"])
    customer_suggestions = top_fuzzy_matches(subject_name, summary_index["all_customers_display"])

    has_certificate_signal = (
        (llm_ok and llm_result.get("is_certificate") is True)
        or (keyword_ok and keyword_result.get("is_certificate") is True)
        or (not llm_ok and not keyword_ok)
    )
    status = "needs_review" if has_certificate_signal else "not_found"
    reason = "No strong match found in certificate_summary.json."
    if status == "needs_review":
        reason = f"{reason} Requiere verificación manual."

    return {
        "status": status,
        "match_type": "none",
        "confidence": 0.0,
        "reason": reason,
        "matches": [],
        "suggestions": {
            "filename": filename_suggestions,
            "customer": customer_suggestions,
        },
        "llm_result": llm_result,
        "keyword_result": keyword_result,
    }


def perform_web_search(query: str, provider: str, api_key: str) -> Dict[str, Any]:
    if not query:
        return {"status": "skipped", "message": "Empty query."}
    if not provider or provider == "none":
        return {"status": "skipped", "message": "Search provider not configured."}
    if not api_key:
        return {"status": "skipped", "message": "API key not provided."}
    return {
        "status": "not_implemented",
        "message": "Web search is not implemented yet. Add provider integration here.",
        "query": query,
    }


def extract_company_name(extraction_result) -> Optional[str]:
    for result in extraction_result.extraction_results:
        if result.success and result.extracted_data and result.extracted_data.company_name:
            return result.extracted_data.company_name
    return None


def run_flow(
    uploaded_files: List[Dict[str, str]],
    intent_inputs: Dict[str, str],
    summary_index: Dict[str, Any],
    catalog_settings: Dict[str, Any],
    notary_inputs: Dict[str, str],
    search_settings: Dict[str, str],
    llm_settings: Dict[str, str],
    content_only: bool,
    verification_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if verification_sources is None:
        verification_sources = []

    intent = CertificateIntentCapture.capture_intent_from_params(
        certificate_type=intent_inputs["certificate_type"],
        purpose=intent_inputs["purpose"],
        subject_name=intent_inputs["subject_name"],
        subject_type=intent_inputs["subject_type"],
        additional_notes=intent_inputs.get("additional_notes") or None,
    )

    requirements = LegalRequirementsEngine.resolve_requirements(intent)

    catalog_path = catalog_settings.get("path")
    catalog_customer = catalog_settings.get("customer")
    catalog_customers = catalog_settings.get("customers", [])
    if not catalog_customer or catalog_customer == "auto":
        catalog_customer = infer_catalog_customer(intent.subject_name, catalog_customers)

    collection = DocumentIntake.create_collection(
        intent,
        requirements,
        catalog_path=catalog_path,
        catalog_customer=catalog_customer,
    )
    file_paths = [item["path"] for item in uploaded_files]
    file_name_overrides = {item["path"]: item["filename"] for item in uploaded_files}
    collection = DocumentIntake.add_files_to_collection(
        collection,
        file_paths,
        file_name_overrides=file_name_overrides,
    )

    extraction = process_collection_with_llm(collection, llm_settings)
    results["phase4"] = extraction.get_summary()

    extracted_company = extract_company_name(extraction)
    subject_name = intent.subject_name.strip() or extracted_company or intent.subject_name
    if subject_name != intent.subject_name:
        intent.subject_name = subject_name

    documents_by_path = {str(doc.file_path): doc for doc in collection.documents}
    extraction_by_path = {
        str(result.document.file_path): result
        for result in extraction.extraction_results
    }

    per_file_data: Dict[str, Dict[str, Any]] = {}
    intent_candidates: List[Dict[str, Any]] = []
    summary_reference = summary_index.get("summary_reference", {})

    for file_info in uploaded_files:
        path = file_info["path"]
        original_filename = file_info["filename"]
        extraction_result = extraction_by_path.get(path)

        doc_text = ""
        if content_only or llm_settings.get("enabled"):
            if (
                extraction_result
                and extraction_result.success
                and extraction_result.extracted_data
            ):
                doc_text = (
                    extraction_result.extracted_data.normalized_text
                    or extraction_result.extracted_data.raw_text
                    or ""
                )

        qa_text = ""
        if extraction_result and extraction_result.success and extraction_result.extracted_data:
            qa_text = extraction_result.extracted_data.raw_text or ""
        if not qa_text:
            qa_text = doc_text
        qa_text = truncate_text(qa_text, MAX_QA_DOC_CHARS)

        llm_result = None
        if llm_settings.get("enabled"):
            if doc_text:
                llm_result = call_groq_classification(
                    model=llm_settings.get("analysis_model", DEFAULT_ANALYSIS_MODEL),
                    api_key=llm_settings.get("api_key", ""),
                    doc_text=doc_text,
                    summary_reference=summary_reference,
                )
            else:
                llm_result = {"status": "error", "message": "No text extracted for LLM."}

        keyword_result = None
        if doc_text:
            keyword_result = keyword_classification(doc_text, summary_reference)

        chosen_classification = choose_classification(llm_result, keyword_result)
        chosen_source = None
        if chosen_classification:
            if llm_result and chosen_classification == llm_result:
                chosen_source = "llm"
            elif keyword_result and chosen_classification == keyword_result:
                chosen_source = "keywords"

        per_file_data[path] = {
            "filename": original_filename,
            "doc_text": doc_text,
            "qa_text": qa_text,
            "llm_result": llm_result,
            "keyword_result": keyword_result,
            "chosen_classification": chosen_classification,
            "chosen_source": chosen_source,
        }

        if chosen_classification:
            intent_candidates.append(
                {
                    "classification": chosen_classification,
                    "source": chosen_source,
                    "confidence": float(chosen_classification.get("confidence", 0.0)),
                    "filename": original_filename,
                }
            )

    intent_override = choose_intent_override(intent_candidates)
    if intent_override:
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type=intent_override["certificate_type"],
            purpose=intent_override["purpose"],
            subject_name=subject_name,
            subject_type=intent.subject_type,
            additional_notes=intent.additional_notes,
        )
        requirements = LegalRequirementsEngine.resolve_requirements(intent)
        collection.certificate_intent = intent
        collection.legal_requirements = requirements
        results["intent_override"] = intent_override

    results["final_subject_name"] = intent.subject_name

    for file_info in uploaded_files:
        path = file_info["path"]
        per_file = per_file_data.get(path, {})
        doc_text = per_file.get("doc_text", "")
        llm_result = per_file.get("llm_result")
        keyword_result = per_file.get("keyword_result")

        match_result = match_document(
            filename=file_info["filename"],
            subject_name=intent.subject_name,
            extracted_company=extracted_company,
            purpose_value=intent.purpose.value,
            summary_index=summary_index,
            llm_result=llm_result,
            keyword_result=keyword_result,
            content_text=doc_text,
            content_only=content_only,
        )
        review_reasons = build_review_reasons(llm_result, keyword_result, match_result)
        per_file["match_result"] = match_result
        per_file["review_required"] = bool(review_reasons)
        per_file["review_reasons"] = review_reasons

    results["phase1"] = intent.get_display_summary()
    results["phase2"] = requirements.get_summary()
    results["phase3"] = collection.get_summary()

    validation = LegalValidator.validate(requirements, extraction)
    results["phase5"] = validation.get_summary()

    gap_report = GapDetector.analyze(validation)
    review_queue = build_review_queue(per_file_data)
    review_gaps = build_review_gaps(review_queue)
    if review_gaps:
        gap_report.gaps.extend(review_gaps)
        gap_report.calculate_summary()
    results["phase6"] = gap_report.get_summary()
    results["gap_structure"] = gap_report.to_dict()

    update_result = DataUpdater.create_update_session(gap_report, collection)
    update_result.updated_extraction_result = extraction
    update_result.review_required = review_queue
    if not verification_sources and (gap_report.gaps or review_queue):
        update_result.system_note = (
            "Fuentes oficiales no configuradas; se requiere verificación manual o carga de documentos."
        )
    results["phase7"] = update_result.get_summary()

    confirmation = FinalConfirmationEngine.confirm(requirements, update_result)
    results["phase8"] = confirmation.get_summary()
    results["confirmation_report"] = confirmation

    validation_by_type = {
        doc_validation.document_type: doc_validation
        for doc_validation in validation.document_validations
        if doc_validation.document_type
    }

    file_results = []
    for file_info in uploaded_files:
        path = file_info["path"]
        original_filename = file_info["filename"]
        doc = documents_by_path.get(path)
        extraction_result = extraction_by_path.get(path)
        per_file = per_file_data.get(path, {})
        catalog_info = doc.metadata.get("catalog") if doc else None
        doc_text = per_file.get("doc_text", "")
        qa_text = per_file.get("qa_text", "")
        llm_result = per_file.get("llm_result")
        keyword_result = per_file.get("keyword_result")
        chosen_classification = per_file.get("chosen_classification")
        match_result = per_file.get("match_result") or {}
        review_required = per_file.get("review_required", False)
        review_reasons = per_file.get("review_reasons", [])

        validation_status = "unknown"
        validation_reason = "No validation available."
        validation_issues = []
        validation_required = None
        detected_doc_type = doc.detected_type if doc else None
        if detected_doc_type and detected_doc_type in validation_by_type:
            doc_validation = validation_by_type[detected_doc_type]
            validation_required = doc_validation.required
            validation_issues = [issue.to_dict() for issue in doc_validation.issues]
            blocking = any(
                issue.severity.value in ("critical", "error")
                for issue in doc_validation.issues
            )
            if not doc_validation.present:
                validation_status = "invalid" if doc_validation.required else "missing_optional"
                validation_reason = "Required document missing." if doc_validation.required else "Optional document missing."
            elif blocking:
                validation_status = "invalid"
                validation_reason = doc_validation.issues[0].description if doc_validation.issues else "Validation errors found."
            else:
                validation_status = "valid"
                validation_reason = "Document passes validation checks."
        elif detected_doc_type:
            validation_status = "not_required"
            validation_reason = "Document type is not required for the selected certificate."

        extraction_error = None
        extraction_warning = None
        llm_extraction_error = None
        text_extraction_error = None
        ocr_error = None
        ocr_used = False
        extraction_success = None
        if extraction_result:
            extraction_success = extraction_result.success
            if not extraction_result.success:
                extraction_error = extraction_result.error
                if validation_status in ("unknown", "not_required"):
                    validation_status = "invalid"
                    validation_reason = f"Extraction failed: {extraction_error}"
            if extraction_result.extracted_data:
                extraction_warning = extraction_result.extracted_data.additional_fields.get(
                    "extraction_warning"
                )
                llm_extraction_error = extraction_result.extracted_data.additional_fields.get(
                    "llm_extraction_error"
                )
                text_extraction_error = extraction_result.extracted_data.additional_fields.get(
                    "text_extraction_error"
                )
                ocr_error = extraction_result.extracted_data.additional_fields.get(
                    "ocr_error"
                )
                ocr_used = bool(
                    extraction_result.extracted_data.additional_fields.get("ocr_used")
                )

        extra_type_hint = detect_pan_card_hint(doc_text)
        doc_type_label = "unknown"
        doc_type_source = "content_missing"
        if llm_result and llm_result.get("status") != "error":
            llm_type = llm_result.get("certificate_type")
            if llm_type:
                doc_type_label = llm_type
                doc_type_source = "content"
        elif keyword_result and keyword_result.get("status") == "ok":
            keyword_type = keyword_result.get("certificate_type")
            if keyword_type:
                doc_type_label = keyword_type
                doc_type_source = "content"
        elif extra_type_hint:
            doc_type_label = extra_type_hint
            doc_type_source = "content"

        detected_doc_type = doc.detected_type.value if doc and doc.detected_type else None
        detected_type_detail = build_detected_type_detail(detected_doc_type, catalog_info)
        file_size_bytes = doc.file_size_bytes if doc else None
        processing_status = doc.processing_status.value if doc else None
        is_scanned = doc.is_scanned if doc else None
        document_type_detail = build_document_type_detail(
            doc_type_label,
            doc_type_source,
            detected_doc_type,
            llm_result,
            keyword_result,
        )
        error_reasons = collect_error_reasons(
            validation_status,
            extraction_success,
            extraction_error,
            llm_extraction_error,
            text_extraction_error,
            ocr_error,
            processing_status,
        )
        has_error = bool(error_reasons)

        file_results.append(
            {
                "filename": original_filename,
                "document_type": doc_type_label,
                "document_type_detected": detected_type_detail,
                "document_type_detail": document_type_detail,
                "type_source": doc_type_source,
                "file_format": doc.file_format.value if doc else "unknown",
                "file_size_bytes": file_size_bytes,
                "processing_status": processing_status,
                "is_scanned": is_scanned,
                "catalog": catalog_info,
                "validation": {
                    "status": validation_status,
                    "reason": validation_reason,
                    "required": validation_required,
                    "issues": validation_issues,
                },
                "match": match_result,
                "llm_result": llm_result,
                "keyword_result": keyword_result,
                "review_required": review_required,
                "review_reasons": review_reasons,
                "doc_text": qa_text,
                "extraction_success": extraction_success,
                "extraction_error": extraction_error,
                "extraction_warning": extraction_warning,
                "llm_extraction_error": llm_extraction_error,
                "text_extraction_error": text_extraction_error,
                "ocr_error": ocr_error,
                "ocr_used": ocr_used,
                "has_error": has_error,
                "error_reasons": error_reasons,
            }
        )

    results["file_results"] = file_results

    if confirmation.can_proceed_to_phase9():
        certificate = CertificateGenerator.generate(
            certificate_intent=intent,
            legal_requirements=requirements,
            extraction_result=update_result.updated_extraction_result,
            confirmation_report=confirmation,
            notary_name=notary_inputs.get("notary_name"),
            notary_office=notary_inputs.get("notary_office"),
        )
        results["phase9"] = certificate.get_summary()
        results["certificate_text"] = certificate.get_formatted_text()

        review_session = NotaryReviewSystem.start_review(
            certificate=certificate,
            reviewer_name=notary_inputs.get("reviewer_name") or "Notary",
        )
        review_session = NotaryReviewSystem.approve_certificate(
            session=review_session,
            notes=notary_inputs.get("review_notes", ""),
        )
        results["phase10"] = review_session.get_summary()

        if review_session.status in [ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CHANGES]:
            final_cert = FinalOutputGenerator.generate_final_certificate(
                certificate=certificate,
                review_session=review_session,
                certificate_number=notary_inputs.get("certificate_number") or "AUTO-0001",
                issuing_notary=notary_inputs.get("notary_name") or "Notary",
                notary_office=notary_inputs.get("notary_office") or "Notary Office",
            )
            results["phase11"] = final_cert.get_summary()
    else:
        results["phase9"] = "Skipped: Phase 8 did not approve certificate generation."
        results["phase10"] = "Skipped: Phase 9 was not generated."
        results["phase11"] = "Skipped: Phase 10 was not approved."

    if search_settings.get("enabled") and file_results:
        has_not_found = any(
            file_result.get("match", {}).get("status") == "not_found"
            for file_result in file_results
        )
        if has_not_found:
            query = f"{intent.subject_name} {intent.purpose.value.replace('para_', '')}"
            search_result = perform_web_search(
                query=query,
                provider=search_settings.get("provider", "none"),
                api_key=search_settings.get("api_key", ""),
            )
            results["web_search"] = search_result

    return results


def render_match_result(match_result: Dict[str, Any]) -> None:
    status = match_result.get("status")
    reason = match_result.get("reason", "")
    confidence = match_result.get("confidence", 0.0)

    if status == "correct":
        st.success(f"Correct: {reason} (confidence {confidence:.2f})")
    elif status == "not_applicable":
        st.info(f"Not applicable: {reason} (confidence {confidence:.2f})")
    elif status == "needs_review":
        st.warning(f"Needs review: {reason} (confidence {confidence:.2f})")
    else:
        st.error(f"Not found: {reason} (confidence {confidence:.2f})")

    matches = match_result.get("matches", [])
    if matches:
        st.write("Matched entries:")
        st.dataframe(matches)

    suggestions = match_result.get("suggestions", {})
    if suggestions:
        if suggestions.get("filename"):
            st.write("Top filename suggestions:")
            st.dataframe(suggestions["filename"], width="stretch")
        if suggestions.get("customer"):
            st.write("Top customer suggestions:")
            st.dataframe(suggestions["customer"], width="stretch")

    llm_result = match_result.get("llm_result")
    if llm_result:
        st.write("LLM classification:")
        st.json(llm_result)
    keyword_result = match_result.get("keyword_result")
    if keyword_result:
        st.write("Keyword classification:")
        st.json(keyword_result)


def format_report_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def format_as_text(data: Any, indent: int = 0, seen=None) -> str:
    """Formats complex data structures into readable text efficiently."""
    if seen is None:
        seen = set()
    
    if not data:
        return ""
    
    # Simple recursion guard (id-based)
    if id(data) in seen:
        return "<circular reference>"
    if isinstance(data, (dict, list)):
        seen.add(id(data))

    prefix = " " * indent
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                formatted_item = format_as_text(item, indent + 2, seen)
                if "\n" in formatted_item:
                    lines.append(f"{prefix}- Item:")
                    lines.append(formatted_item)
                else:
                    lines.append(f"{prefix}- {formatted_item.strip()}")
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            label = str(k).replace("_", " ").title()
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{label}:")
                lines.append(format_as_text(v, indent + 2, seen))
            else:
                # Truncate very long strings in the report for performance
                val_str = str(v)
                if len(val_str) > 1000:
                    val_str = val_str[:1000] + "... [truncated]"
                lines.append(f"{prefix}{label}: {val_str}")
        return "\n".join(lines)
    
    return f"{prefix}{str(data)}"


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    st.set_page_config(page_title="Notarial Chatbot Flow", layout="wide")
    st.title("Notarial Chatbot Flow")
    st.write("Streamlit UI for phases 1-11, dataset matching, and optional web search stub.")
    st.info("Note: LLMs read text only. Enable OCR fallback for scanned PDFs/images.")

    st.sidebar.header("Settings")
    with st.sidebar.expander("Dataset settings", expanded=False):
        summary_path = st.text_input("certificate_summary.json path", DEFAULT_SUMMARY_PATH)
        catalog_path = st.text_input("client_file_catalogs.json path", DEFAULT_CATALOG_PATH)
    enable_llm = st.sidebar.checkbox("Enable LLM extraction + classification (Groq)", value=True)
    enable_ocr_fallback = st.sidebar.checkbox(
        "Enable OCR fallback when no text is found",
        value=True,
    )
    content_only = st.sidebar.checkbox("Match by content only", value=True)
    if enable_llm:
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        extraction_model = st.sidebar.text_input(
            "Groq extraction model",
            value=DEFAULT_EXTRACTION_MODEL,
        )
        analysis_model = st.sidebar.text_input(
            "Groq analysis model",
            value=DEFAULT_ANALYSIS_MODEL,
        )
        qa_model = st.sidebar.text_input(
            "Groq Q&A model",
            value=DEFAULT_QA_MODEL,
        )
        if groq_api_key:
            st.sidebar.caption("GROQ API key loaded from .env")
        else:
            st.sidebar.warning("GROQ API key not found in .env")
    else:
        groq_api_key = ""
        extraction_model = DEFAULT_EXTRACTION_MODEL
        analysis_model = DEFAULT_ANALYSIS_MODEL
        qa_model = DEFAULT_QA_MODEL
    enable_search = st.sidebar.checkbox("Enable web search fallback (stub)", value=False)
    if enable_search:
        search_provider = st.sidebar.selectbox("Search provider", ["none", "serpapi", "bing"])
        search_api_key = st.sidebar.text_input("API key", type="password")
    else:
        search_provider = "none"
        search_api_key = ""

    summary_path_obj = Path(summary_path)
    if not summary_path_obj.exists():
        st.error(f"Summary file not found: {summary_path}")
        st.stop()

    summary_data = load_summary(summary_path)
    summary_index = build_summary_index(summary_data)
    summary_index["summary_reference"] = build_llm_reference(summary_data)

    catalog_data = load_client_catalog(catalog_path)
    catalog_customers = sorted(catalog_data.keys())
    if catalog_customers:
        catalog_customer = st.sidebar.selectbox(
            "Catalog customer (optional)",
            options=["auto"] + catalog_customers,
            index=0,
        )
    else:
        catalog_customer = "auto"
        if catalog_path:
            st.sidebar.info("Catalog file not found or empty.")

    st.sidebar.markdown("### Summary stats")
    st.sidebar.write(f"Total entries: {len(summary_index['entries'])}")
    st.sidebar.write(f"Certificates: {len([e for e in summary_index['entries'] if e['entry_type'] == 'certificate'])}")
    st.sidebar.write(f"Non-certificates: {len([e for e in summary_index['entries'] if e['entry_type'] == 'non_certificate'])}")
    if catalog_customers:
        st.sidebar.write(f"Catalog customers: {len(catalog_customers)}")

    st.subheader("Inputs")
    cert_types = CertificateIntentCapture.get_available_certificate_types()
    purposes = CertificateIntentCapture.get_available_purposes()
    default_cert_type = get_default_option(cert_types, DEFAULT_CERT_TYPE)
    default_purpose = get_default_option(purposes, DEFAULT_PURPOSE)

    simple_mode = st.checkbox("Simple mode (use defaults)", value=True)
    if simple_mode:
        st.caption(
            f"Defaults: {default_cert_type['label']} / {default_purpose['label']} / subject type = company."
        )

    cert_type = default_cert_type
    purpose = default_purpose
    subject_type = "company"
    additional_notes = ""
    notary_name = "Dr. Juan Perez"
    notary_office = "Notary Office"
    reviewer_name = "Dr. Juan Perez"
    certificate_number = "AUTO-0001"
    review_notes = "Auto-approved (demo)"

    input_method = st.radio("Input method", ["Upload folder (select multiple files)", "Local Folder Path"])

    with st.form("flow_form"):
        uploaded_files = []
        folder_path = ""
        
        if input_method == "Upload folder (select multiple files)":
            uploaded_files = st.file_uploader(
                "Upload folder (select multiple files)",
                type=None,
                accept_multiple_files=True,
            )
        else:
            folder_path = st.text_input("Local Folder Path (Absolute path)", value="")
            st.caption("Example: /home/user/documents/client_files")

        subject_name = st.text_input("Subject name (optional)", value="")

        if not simple_mode:
            with st.expander("Certificate details", expanded=True):
                cert_type_index = next(
                    (idx for idx, item in enumerate(cert_types) if item["value"] == default_cert_type["value"]),
                    0,
                )
                purpose_index = next(
                    (idx for idx, item in enumerate(purposes) if item["value"] == default_purpose["value"]),
                    0,
                )
                col1, col2 = st.columns(2)
                with col1:
                    cert_type = st.selectbox(
                        "Certificate type",
                        options=cert_types,
                        index=cert_type_index,
                        format_func=lambda x: x["label"],
                    )
                with col2:
                    purpose = st.selectbox(
                        "Purpose",
                        options=purposes,
                        index=purpose_index,
                        format_func=lambda x: x["label"],
                    )
                subject_type = st.selectbox("Subject type", ["company", "person"])
                additional_notes = st.text_input("Additional notes (optional)", value="")

            with st.expander("Notary info", expanded=False):
                notary_name = st.text_input("Notary name", value=notary_name)
                notary_office = st.text_input("Notary office", value=notary_office)
                reviewer_name = st.text_input("Reviewer name", value=reviewer_name)
                certificate_number = st.text_input("Certificate number", value=certificate_number)
                review_notes = st.text_input("Review notes", value=review_notes)

        submit = st.form_submit_button("Run flow")

    results = st.session_state.get("last_results")
    if submit:
        tmp_dir = Path(".tmp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        tmp_paths = []
        uploaded_items = []

        if input_method == "Upload folder (select multiple files)":
            if not uploaded_files:
                st.error("Please upload documents before running the flow.")
                return

            for uploaded_file in uploaded_files:
                tmp_filename = f"{uuid4().hex}_{uploaded_file.name}"
                tmp_path = tmp_dir / tmp_filename
                tmp_path.write_bytes(uploaded_file.getbuffer())
                tmp_paths.append(tmp_path)
                uploaded_items.append(
                    {
                        "path": str(tmp_path),
                        "filename": uploaded_file.name,
                    }
                )
        else:
            if not folder_path:
                st.error("Please provide a local folder path.")
                return
            
            # Sanitize path: remove potential terminal prompt copy-paste artifacts
            sanitized_path = folder_path.strip()
            # Remove trailing $ or %
            if sanitized_path.endswith("$") or sanitized_path.endswith("%"):
                sanitized_path = sanitized_path[:-1].strip()
            # Remove user@host: prefix if present
            if "@" in sanitized_path and ":" in sanitized_path:
                 # Split on the *first* colon, which separates host from path in standard prompts
                 parts = sanitized_path.split(":", 1)
                 # Only take the right side if it looks like a path (starts with / or ~)
                 if len(parts) > 1 and (parts[1].strip().startswith("/") or parts[1].strip().startswith("~")):
                     sanitized_path = parts[1].strip()

            p = Path(sanitized_path).expanduser()
            
            if not p.exists() or not p.is_dir():
                st.error(f"Invalid folder path: {sanitized_path} (Raw input: {folder_path})")
                return
            
            # List files in the directory
            files_found = [f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]
            if not files_found:
                 st.error(f"No files found in {folder_path}")
                 return
            
            st.success(f"Found {len(files_found)} files in {folder_path}")
            
            for file_path in files_found:
                 uploaded_items.append(
                    {
                        "path": str(file_path.absolute()),
                        "filename": file_path.name,
                    }
                )

        intent_inputs = {
            "certificate_type": cert_type["value"],
            "purpose": purpose["value"],
            "subject_name": subject_name.strip(),
            "subject_type": subject_type,
            "additional_notes": additional_notes.strip(),
        }
        notary_inputs = {
            "notary_name": notary_name.strip(),
            "notary_office": notary_office.strip(),
            "reviewer_name": reviewer_name.strip(),
            "certificate_number": certificate_number.strip(),
            "review_notes": review_notes.strip(),
        }
        search_settings = {
            "enabled": enable_search,
            "provider": search_provider,
            "api_key": search_api_key,
        }
        llm_settings = {
            "enabled": enable_llm,
            "extraction_model": extraction_model,
            "analysis_model": analysis_model,
            "api_key": groq_api_key,
            "ocr_fallback": enable_ocr_fallback,
        }
        catalog_settings = {
            "path": catalog_path,
            "customer": catalog_customer,
            "customers": catalog_customers,
            "data": catalog_data,
        }

        try:
            with st.spinner("Analyzing documents and generating reports..."):
                results = run_flow(
                    uploaded_files=uploaded_items,
                    intent_inputs=intent_inputs,
                    summary_index=summary_index,
                    catalog_settings=catalog_settings,
                    notary_inputs=notary_inputs,
                    search_settings=search_settings,
                    llm_settings=llm_settings,
                    content_only=content_only,
                )
                st.session_state["last_results"] = results
                st.session_state["qa_history"] = []
                
                # Pre-generate reports immediately after flow completes
                if results and "file_results" in results:
                     # Calculate Summary Header
                    subject_final = results.get("final_subject_name") or subject_name or "Unknown Subject"
                    gap_struct = results.get("gap_structure") or {}
                    gaps_list = gap_struct.get("gaps", [])
                    total_files = len(results["file_results"])
                    error_count = sum(1 for f in results["file_results"] if f.get("has_error"))
                    doc_types_found = sorted(list({f.get("document_type", "unknown") for f in results["file_results"]}))
                    doc_types_str = ", ".join(doc_types_found) if doc_types_found else "None"
                    missing_docs = [gap.get("title", "").replace("Falta ", "") for gap in gaps_list if gap.get("gap_type") == "missing_document" and gap.get("priority") == "urgent"]
                    missing_docs_str = ", ".join(missing_docs) if missing_docs else "None"
                    status_overall = "ATTENTION REQUIRED" if (error_count > 0 or missing_docs) else "VALID"

                    summary_header = [
                        "SUMMARY OF ANALYSIS",
                        "-" * 20,
                        f"Subject: {subject_final}",
                        f"Total Files Processed: {total_files}",
                        f"Documents Found: {doc_types_str}",
                        f"Status: {status_overall}",
                        "",
                        f"Errors Found: {'Yes' if error_count > 0 else 'No'} ({error_count} files affected)",
                        f"Missing Required Documents: {missing_docs_str}",
                        "=" * 60,
                        "",
                    ]

                    detailed_lines = ["Notarial Chatbot Flow - Detailed Analysis Report", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "=" * 60] + summary_header
                    short_lines = ["Notarial Chatbot Flow - Short Summary", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "=" * 60] + summary_header
                    
                    for fr in results["file_results"]:
                        # Short entry
                        short_lines.extend([
                            f"File: {fr.get('filename')}",
                            f"Subject: {subject_final}",
                            f"Type: {fr.get('document_type')}",
                            f"Status: {str(fr.get('validation', {}).get('status', 'unknown')).upper()}",
                            f"Errors: {'Yes' if fr.get('has_error') else 'No'}",
                            "-" * 20
                        ])
                        # Detailed entry
                        detailed_lines.append(f"Details: {fr.get('filename')}")
                        detailed_lines.append(f"Type: {fr.get('document_type')} (source: {fr.get('type_source')})")
                        detailed_lines.append(f"Validation: {fr.get('validation', {}).get('status')} - {fr.get('validation', {}).get('reason')}")
                        if fr.get("has_error"):
                             detailed_lines.append(f"Errors: {format_as_text(fr.get('error_reasons'), 2)}")
                        if fr.get("validation", {}).get("issues"):
                             detailed_lines.append(f"Issues: {format_as_text(fr.get('validation', {}).get('issues'), 2)}")
                        detailed_lines.append("-" * 80)
                    
                    st.session_state["detailed_report_text"] = "\n".join(detailed_lines)
                    st.session_state["short_summary_text"] = "\n".join(short_lines)
                    
                    # Also write to local files once
                    (output_dir / "notary_summary.txt").write_text(st.session_state["short_summary_text"], encoding="utf-8")
                    (output_dir / "notary_detailed_report.txt").write_text(st.session_state["detailed_report_text"], encoding="utf-8")

        except Exception as exc:
            st.exception(exc)
            return
        finally:
            for tmp_path in tmp_paths:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            try:
                if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                    tmp_dir.rmdir()
            except OSError:
                pass
    if results is None:
        return

    st.subheader("Document analysis")
    file_results = results.get("file_results", [])
    if file_results:
        summary_rows = []
        report_rows = []
        review_files = [fr for fr in file_results if fr.get("review_required")]
        if review_files:
            st.warning(f"{len(review_files)} documento(s) requieren revisión manual.")
        for file_result in file_results:
            validation = file_result.get("validation", {})
            match_result = file_result.get("match") or {}
            llm_result = file_result.get("llm_result") or {}
            keyword_result = file_result.get("keyword_result") or {}
            catalog = file_result.get("catalog") or {}
            summary_rows.append(
                {
                    "file": file_result.get("filename"),
                    "document_type": file_result.get("document_type"),
                    "document_type_detected": file_result.get("document_type_detected"),
                    "type_source": file_result.get("type_source"),
                    "catalog_match": catalog.get("match_status"),
                    "has_error": format_has_error_flag(file_result.get("has_error")),
                }
            )
            report_rows.append(
                {
                    "file": file_result.get("filename"),
                    "document_type": file_result.get("document_type"),
                    "document_type_detected": file_result.get("document_type_detected"),
                    "document_type_detail": file_result.get("document_type_detail"),
                    "type_source": file_result.get("type_source"),
                    "file_format": file_result.get("file_format"),
                    "file_size_bytes": file_result.get("file_size_bytes"),
                    "processing_status": file_result.get("processing_status"),
                    "is_scanned": file_result.get("is_scanned"),
                    "extraction_warning": file_result.get("extraction_warning"),
                    "extraction_error": file_result.get("extraction_error"),
                    "text_extraction_error": file_result.get("text_extraction_error"),
                    "ocr_error": file_result.get("ocr_error"),
                    "ocr_used": file_result.get("ocr_used"),
                    "has_error": format_has_error_detail(
                        file_result.get("has_error"),
                        file_result.get("error_reasons") or [],
                    ),
                    "match_status": match_result.get("status"),
                    "match_type": match_result.get("match_type"),
                    "match_reason": match_result.get("reason"),
                    "llm_status": llm_result.get("status"),
                    "llm_is_certificate": llm_result.get("is_certificate"),
                    "llm_certificate_type": llm_result.get("certificate_type"),
                    "llm_purpose": llm_result.get("purpose"),
                    "llm_confidence": llm_result.get("confidence"),
                    "keyword_status": keyword_result.get("status"),
                    "keyword_is_certificate": keyword_result.get("is_certificate"),
                    "keyword_certificate_type": keyword_result.get("certificate_type"),
                    "keyword_purpose": keyword_result.get("purpose"),
                    "keyword_confidence": keyword_result.get("confidence"),
                    "catalog_customer": catalog.get("customer"),
                    "catalog_source": catalog.get("source_file"),
                    "catalog_description": catalog.get("description"),
                    "catalog_expected_extensions": ",".join(catalog.get("expected_extensions", []) or []),
                    "catalog_match_status": catalog.get("match_status"),
                    "catalog_match_score": catalog.get("match_score"),
                    "catalog_type_mismatch": catalog.get("type_mismatch"),
                    "review_required": file_result.get("review_required"),
                    "review_reasons": "; ".join(file_result.get("review_reasons") or []),
                }
            )
        st.dataframe(summary_rows, width="stretch")
        if report_rows:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
            csv_text = output.getvalue()
            save_report = st.download_button(
                "Download file analysis report (CSV)",
                csv_text.encode("utf-8"),
                file_name="notary_file_report.csv",
                mime="text/csv",
            )
            if save_report:
                (output_dir / "notary_file_report.csv").write_text(
                    csv_text,
                    encoding="utf-8",
                )
        short_summary_report = st.session_state.get("short_summary_text", "")
        detailed_report = st.session_state.get("detailed_report_text", "")

        if short_summary_report and detailed_report:
            col1, col2 = st.columns(2)
            with col1:
                 st.download_button(
                    "Download Short Summary (TXT)",
                    short_summary_report.encode("utf-8"),
                    file_name="notary_summary.txt",
                    mime="text/plain",
                )
            with col2:
                 st.download_button(
                    "Download Detailed Report (TXT)",
                    detailed_report.encode("utf-8"),
                    file_name="notary_detailed_report.txt",
                    mime="text/plain",
                )

        for file_result in file_results:
            filename = file_result.get("filename", "document")
            with st.expander(f"Details: {filename}", expanded=False):
                st.write(
                    "Type: "
                    f"{file_result.get('document_type')} "
                    f"(source: {file_result.get('type_source')})"
                )
                st.write(
                    "Detected type (auto): "
                    f"{file_result.get('document_type_detected')}"
                )
                st.write(
                    "File info: "
                    f"format={file_result.get('file_format')}, "
                    f"size_bytes={file_result.get('file_size_bytes')}, "
                    f"processing_status={file_result.get('processing_status')}, "
                    f"is_scanned={file_result.get('is_scanned')}"
                )
                if file_result.get("has_error") is not None:
                    st.write(
                        f"Error flag: {'yes' if file_result.get('has_error') else 'no'}"
                    )
                    if file_result.get("error_reasons"):
                        st.write("Error reasons:")
                        st.write("; ".join(file_result.get("error_reasons") or []))
                validation = file_result.get("validation", {})
                st.write(
                    "Validation: "
                    f"{validation.get('status')} - {validation.get('reason')}"
                )
                if file_result.get("extraction_error"):
                    st.warning(f"Extraction error: {file_result['extraction_error']}")
                if file_result.get("extraction_warning"):
                    st.warning(f"Extraction warning: {file_result['extraction_warning']}")
                if file_result.get("llm_extraction_error"):
                    st.warning(f"LLM extraction error: {file_result['llm_extraction_error']}")
                if file_result.get("text_extraction_error"):
                    st.warning(f"Text extraction error: {file_result['text_extraction_error']}")
                if file_result.get("ocr_error"):
                    st.warning(f"OCR error: {file_result['ocr_error']}")
                if file_result.get("ocr_used"):
                    st.info("OCR was used to extract text for this file.")
                if validation.get("issues"):
                    st.write("Validation issues:")
                    st.dataframe(validation["issues"], width="stretch")
                match_result = file_result.get("match")
                if match_result:
                    st.write("Dataset match:")
                    render_match_result(match_result)
                if file_result.get("llm_result"):
                    st.write("LLM classification:")
                    st.json(file_result["llm_result"])
                if file_result.get("keyword_result"):
                    st.write("Keyword classification:")
                    st.json(file_result["keyword_result"])
    else:
        st.info("No documents were processed.")

    if results.get("intent_override"):
        override = results["intent_override"]
        file_label = f", file: {override.get('filename')}" if override.get("filename") else ""
        st.info(
            "Intent overridden from content: "
            f"{override['certificate_type']} / {override['purpose']} "
            f"(source: {override['source']}{file_label}, confidence: {override['confidence']:.2f})"
        )

    if results.get("web_search"):
        st.subheader("Web search fallback")
        st.write(results["web_search"])

    st.subheader("Phase outputs")
    phase_sections = [
        ("Phase 1", "phase1"),
        ("Phase 2", "phase2"),
        ("Phase 3", "phase3"),
        ("Phase 4", "phase4"),
        ("Phase 5", "phase5"),
        ("Phase 6", "phase6"),
        ("Phase 7", "phase7"),
        ("Phase 8", "phase8"),
        ("Phase 9", "phase9"),
        ("Phase 10", "phase10"),
        ("Phase 11", "phase11"),
    ]

    phase_lines = [
        "Notarial Chatbot Flow - Phase Outputs",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for label, key in phase_sections:
        phase_lines.append(label)
        phase_lines.append(str(results.get(key, "No output")))
        phase_lines.append("-" * 80)
    phase_output_text = "\n".join(phase_lines)
    save_phases = st.download_button(
        "Download phase outputs (TXT)",
        phase_output_text.encode("utf-8"),
        file_name="notary_phase_outputs.txt",
        mime="text/plain",
    )
    if save_phases:
        (output_dir / "notary_phase_outputs.txt").write_text(
            phase_output_text,
            encoding="utf-8",
        )

    for label, key in phase_sections:
        with st.expander(label, expanded=False):
            st.code(str(results.get(key, "No output")), language="text")

    if results.get("certificate_text"):
        st.subheader("Generated certificate text")
        st.code(results["certificate_text"], language="text")

    st.subheader("Document Q&A")
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    file_results = results.get("file_results", [])
    certificate_text = results.get("certificate_text", "")
    if not file_results and not certificate_text:
        st.info("No document content available for Q&A.")
    elif not enable_llm:
        st.info("Enable LLM extraction + classification to use Q&A.")
    elif not groq_api_key:
        st.warning("GROQ API key not found in .env.")
    else:
        scope_options = ["All documents"]
        scope_map = {"All documents": "all"}
        if certificate_text:
            scope_options.append("Generated certificate")
            scope_map["Generated certificate"] = "certificate"
        for idx, file_result in enumerate(file_results):
            label = f"File {idx + 1}: {file_result.get('filename', 'document')}"
            scope_options.append(label)
            scope_map[label] = f"file:{idx}"

        with st.form("qa_form"):
            scope_label = st.selectbox("Question scope", options=scope_options)
            question = st.text_area("Question", value="")
            submitted = st.form_submit_button("Ask")

        if submitted:
            context = build_qa_context(
                file_results=file_results,
                certificate_text=certificate_text,
                scope_key=scope_map.get(scope_label, "all"),
            )
            qa_result = call_groq_document_qa(
                model=qa_model,
                api_key=groq_api_key,
                question=question,
                context=context,
            )
            if qa_result.get("status") == "ok":
                st.session_state["qa_history"].append(
                    {
                        "question": question.strip(),
                        "answer": qa_result.get("answer", ""),
                        "scope": scope_label,
                    }
                )
            else:
                st.error(qa_result.get("message", "Q&A failed."))

        if st.session_state["qa_history"]:
            st.write("Conversation")
            for entry in st.session_state["qa_history"][-5:]:
                st.write(f"Q ({entry.get('scope')}): {entry.get('question')}")
                st.write(f"A: {entry.get('answer')}")

        if st.button("Clear Q&A history"):
            st.session_state["qa_history"] = []


if __name__ == "__main__":
    main()
