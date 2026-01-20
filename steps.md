# Project Status and Notes

## What the code is doing right now
- Runs an 11-phase pipeline end to end to classify documents, validate legal requirements, detect gaps, and generate a certificate draft when allowed.
- Uses LLM + keyword matching on document content (not filenames) to classify document types and purposes.
- Uses `certificate_summary.json` as a local taxonomy and matching dataset; if a document cannot be matched, it is marked for manual review.
- Phase 7 (official verification/update) is a stub; no external APIs are connected yet.
- Saves the three downloadable reports next to `chatbot.py` when the download buttons are clicked.

## Simple flow (human language)
1) The notary chooses certificate type and purpose.
2) The notary uploads documents.
3) The system reads the files, extracts text, and tries to classify them.
4) The system checks legal requirements (missing docs, missing fields, expired docs, article compliance).
5) The system produces a gap/fix list.
6) If updates are needed, it asks for manual upload; official sources are not connected yet.
7) If everything is valid, it generates a certificate draft and asks for notary review.
8) If not valid, it blocks generation and reports what must be fixed.

## Manual review behavior (because dataset is incomplete)
- If LLM and keywords disagree, or the dataset has no match, the document is marked for manual review.
- Review items appear in Phase 6 as urgent gaps and in Phase 7 as a review queue.
- Phase 8 will require manual review before approval when review items exist.

## Files and what each one does

### Entry point and UI
- `chatbot.py`: Streamlit UI and main orchestrator; runs phases 1-11, builds reports, and saves downloads.

### Phases (core pipeline)
- `src/phase1_certificate_intent.py`: Captures user intent (certificate type, purpose, subject).
- `src/phase2_legal_requirements.py`: Builds legal requirements, required documents, and articles.
- `src/phase3_document_intake.py`: Ingests uploads, detects document types (heuristic), and builds the collection.
- `src/phase4_text_extraction.py`: Extracts text from files (PDF/DOC/DOCX/images) and normalizes it.
- `src/phase5_legal_validation.py`: Validates documents, elements, expiry, cross-document consistency, and article compliance.
- `src/phase6_gap_detection.py`: Converts validation results into gap reports and action steps.
- `src/phase7_data_update.py`: Placeholder for updates (manual upload or public registry fetch in the future).
- `src/phase8_final_confirmation.py`: Re-validates after updates and decides approve/reject/needs review.
- `src/phase9_certificate_generation.py`: Generates the certificate draft text.
- `src/phase10_notary_review.py`: Tracks notary review and edits.
- `src/phase11_final_output.py`: Produces the final certificate output.

### Data and references
- `cetificate from dataset/certificate_summary.json`: Local taxonomy and matching dataset used for classification.
- `cetificate from dataset/client_file_catalogs.json`: Optional client catalog for matching file expectations.
- `Notaria_client_data/`: Source dataset used to build the summary file.
- `articles/`: Local article excerpts used for legal compliance checks.

### Outputs (generated per run)
- `notary_file_report.csv`: One-line-per-file report for quick review.
- `notary_file_details.txt`: Full per-file details and match/classification output.
- `notary_phase_outputs.txt`: Full phase summaries (1-11).

### Project support files
- `requirements.txt`: Python dependencies.
- `README.md`: Project overview and usage notes.
- `workflow.md`: High-level flow description.
- `tests/`: Tests (if any are added/running).
- `.env`: API keys (Groq), if configured.
