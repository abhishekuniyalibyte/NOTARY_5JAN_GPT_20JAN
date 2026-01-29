# Project Status and Notes

## Workflow Overview (Runtime)
- The system runs an 11-phase pipeline from intent capture to final certificate output.
- Document content (not just filenames) is extracted, normalized, and classified.
- Legal validation is rule-driven using Articles 248-255 and cross-references.
- Gaps and missing items are surfaced before certificate generation is allowed.

## Phase-by-Phase Workflow
1) Phase 1: Capture certificate intent (type, purpose, subject).
2) Phase 2: Resolve legal requirements and required documents.
3) Phase 3: Intake documents and detect probable types.
4) Phase 4: Extract text/OCR and structure key fields.
5) Phase 5: Validate against legal requirements and document rules.
6) Phase 6: Convert validation results into gaps and actions.
7) Phase 7: Data update attempts (currently a stub/manual).
8) Phase 8: Final confirmation gate (approve/reject/needs review).
9) Phase 9: Generate certificate draft text.
10) Phase 10: Capture notary review feedback.
11) Phase 11: Produce final output package.

## Manual Review Behavior
- If LLM and keyword classification disagree, or no match exists in the dataset,
  the document is marked for manual review.
- Review items appear in Phase 6 (gaps) and Phase 7 (review queue).
- Phase 8 blocks approval until review items are resolved.

## Files and Directories (What Each Does)

### Entry Point and UI
- `chatbot.py`: Streamlit UI and main orchestrator for phases 1-11.
- `other files/`: Alternate UI/prototype variants kept for reference.

### Core Pipeline (src/)
- `src/phase1_certificate_intent.py`: Captures certificate type, purpose, subject.
- `src/phase2_legal_requirements.py`: Rules engine for legal requirements.
- `src/phase3_document_intake.py`: Ingests uploads and basic type detection.
- `src/phase4_text_extraction.py`: OCR/text extraction and field normalization.
- `src/phase5_legal_validation.py`: Legal validation against rules/articles.
- `src/phase6_gap_detection.py`: Gap analysis and action prioritization.
- `src/phase7_data_update.py`: Update mechanism (placeholder for APIs).
- `src/phase8_final_confirmation.py`: Final decision gate.
- `src/phase9_certificate_generation.py`: Certificate draft generation.
- `src/phase10_notary_review.py`: Tracks notary feedback.
- `src/phase11_final_output.py`: Final output packaging.

### Data and Knowledge Sources
- `articles/`: Local article excerpts for legal checks.
- `Notaria_client_data/`: Client-provided historical documents.
- `cetificate from dataset/`: Derived datasets and scripts:
  - `certificate_summary.json`: Taxonomy used for classification.
  - `client_file_catalogs.json`: Optional client catalog hints.
  - `certificate_types.json`: Normalized certificate type listing.
  - `customers_index.json`: Index of customer files.
  - `certificate_summary.py` and `text_extractor.py`: Preprocessing utilities.

### Reports (Generated Per Run)
- `notary_file_report.csv`: One-line per file summary.
- `notary_file_details.txt`: Detailed classification output.
- `notary_phase_outputs.txt`: Full phase logs.

### Docs and Requirements
- `README.md`: Project overview and usage.
- `workflow.md`: High-level workflow description.
- `steps.md`: Working notes and project status.
- `files_description.md`: Quick inventory of key files.
- `client_requirements.txt`: Client scope and requirements.

### Tests and Tooling
- `tests/`: Unit tests for each phase.
- `requirements.txt`: Python dependencies.
- `.env`: API keys (if configured).
- `.gitignore`: Git ignore rules.
- `venv/`: Local virtual environment (if present).

## Next Steps (Recommended)
- Add a custom RAG layer: ingest legal articles and historical certificates,
  store embeddings with metadata, and retrieve context for LLM classification
  and drafting with citations.
- Wire retrieval into `chatbot.py` so Phase 4/9 can use grounded context,
  while Phase 5 remains deterministic rule validation.
- Implement a retrieval evaluation checklist (top-k recall, confidence gating)
  and route low-confidence matches to manual review.
- Connect Phase 7 to real update sources (registry/APIs) or formalize the
  manual update workflow if external integration is deferred.
- Introduce template storage/versioning per notary/institution and feed it
  into Phase 9 generation and Phase 10 feedback loops.
