NOTARIA PROJECT README

1. What this project does

This project helps a notary upload a set of documents, extract the important information from them, validate the set against predefined legal and institutional requirements, and only generate a certificate draft when the requirements are satisfied. The main goal is to make it very clear what is missing, what is inconsistent, what is expired, and what needs manual verification before proceeding.


2. End-to-end workflow (from upload to outputs)

The user starts in the Streamlit app and chooses the certificate type and purpose (for example a signature certification for BPS). The user can also enter the subject (company/person name). The user uploads multiple files or provides a local folder path. For uploads, the app saves each file into a temporary local directory so the rest of the pipeline can work with file paths while still preserving the original filenames for reporting.
After the files are available locally, the system ingests them and creates an internal document collection. Each file is assigned basic metadata (format, size, processing status), and the system tries to detect the document type using filename keywords. If a client catalog is selected (loaded from the JSON that was created from a client Excel file such as ACRISOUND.xlsx), the system also tries to match uploaded filenames to catalog entries and attaches catalog metadata (description, expected formats, match method/status). This catalog is mainly for hints and reporting; it does not define the legal requirements.
Next, the system extracts text from each document. For PDFs/DOCX/TXT it tries direct extraction. For legacy DOC it may use LibreOffice conversion or other fallbacks. If no text is extracted and OCR fallback is enabled, it attempts OCR for scanned PDFs/images. The extracted text is normalized so later steps can work more consistently.
From the extracted text, the system tries to extract structured fields (company name, RUT, CI, registry references, dates, etc.). Regex extraction is always used as a fallback. If Groq is enabled and an API key is present, the system can also ask the LLM to extract fields and to classify the document based on its content.
Independently from extraction, the system builds a requirement checklist based on the selected certificate type and purpose. This checklist is defined in code (it is not being inferred from the dataset). It includes which document types are mandatory, which are optional, which have expiry windows (for example 30 days), and the legal basis for each requirement.
The system then validates the uploaded set against the checklist. It checks whether required documents are present, whether required fields are found inside the relevant documents, whether documents are too old when expiry rules apply, whether the data is consistent across the entire set (for example company name and RUT should match), and whether the extraction quality is good enough to trust the results. The system also produces an article compliance section by mapping legal articles to required elements and optionally attaching a short excerpt of the article text from the local articles folder. The excerpt is for context; the enforcement is still rule-based.
After validation, the system converts validation problems into a set of actionable gaps (missing documents, inconsistencies, review-needed items). If external official sources are not configured, the system cannot fetch missing documents automatically. In that situation it keeps the gaps and directs the user to upload or manually verify the missing items.
Finally, the system runs a final confirmation gate. If blocking issues remain (missing mandatory documents, critical validation problems, unresolved urgent gaps), certificate generation is rejected and later steps are skipped. If the gate passes, the system generates a certificate draft, runs a review step, and produces final outputs.
At the end of every run, the system writes reports that summarize what happened. The short summary is meant for quick status, the detailed report is meant for explaining the reasons and the legal basis behind each issue, and the CSV is meant for spreadsheet-style review of each uploaded file.


3. What defines “which documents are required”

The required-document checklist is defined in src/phase2_legal_requirements.py. It maps certificate type and purpose to:
1) required document types (mandatory vs optional)
2) expiry rules (expires, expiry_days)
3) institution-specific rules (such as BPS validity days and additional requirements)
4) legal basis references (article/law labels used in reports)

This is the main place to change if a client wants different requirements or if the legal requirements change.


4. What the dataset JSON files are used for

cetificate from dataset/certificate_summary.json
This is used as a taxonomy and reference for classification and dataset-style matching. It helps the system label documents by type/purpose and provides a structured set of known categories, attributes, and examples. It is not used to decide what is legally required for a specific certificate request.

cetificate from dataset/customers_index.json
This is an index of historical files grouped by customer. It is mainly a preprocessing artifact used to build certificate_summary.json and other derived summaries. The Streamlit runtime workflow does not use customers_index.json directly.

cetificate from dataset/client_file_catalogs.json
This is a client-specific catalog created from a client-provided Excel file (for example ACRISOUND.xlsx). At runtime it is used to match uploaded filenames to expected client documents so the system can attach descriptions and expected formats, and warn when the uploaded file does not match what the catalog expects. If files are uploaded with generic names like 1.pdf, this catalog matching will usually not work unless the user renames files or you extend the catalog logic to match by content.


5. What the articles folder is used for

The articles folder contains local text files for legal articles. The current code uses these texts for short excerpts in the reporting layer so the notary/client can see context. The rule enforcement is not based on parsing the article text; it is based on explicit checks and mappings in code.


6. File and folder guide (what each part does)

Entry points and UI

chatbot.py
Main Streamlit UI and orchestrator. Handles upload, settings, runs the full workflow, and writes notary_summary.txt, notary_detailed_report.txt, notary_file_report.csv, and notary_phase_outputs.txt.

chatbot_openai.py, chatbot_copy.py
Older or alternate variants kept for reference.

Core logic (src)

src/phase1_certificate_intent.py
Captures the request intent (certificate type, purpose, subject name/type).

src/phase2_legal_requirements.py
Defines the rule mappings for required documents, expiry windows, and institution rules.

src/phase3_document_intake.py
Ingests files, detects probable document types, attaches client catalog metadata when configured.

src/phase4_text_extraction.py
Text extraction, OCR support, normalization, and extracted field structures.

src/phase5_legal_validation.py
Checks required-document presence, expiry, required fields, quality, consistency, and produces article-based compliance checks (rule-based, with optional excerpts).

src/phase6_gap_detection.py
Turns validation results into actionable gaps and priorities.

src/phase7_data_update.py
Creates an update session. In the current setup it is mostly manual and does not fetch official documents automatically.

src/phase8_final_confirmation.py
Final go/no-go decision. Blocks certificate generation if blocking issues remain.

src/phase9_certificate_generation.py
Builds a certificate draft from the validated data and rules.

src/phase10_notary_review.py
Tracks review status and notary feedback.

src/phase11_final_output.py
Final output packaging/export step.

Data and preprocessing scripts

cetificate from dataset/
Contains derived JSONs and helper scripts used to build summaries and client catalogs from historical data.

articles/
Local article excerpts used for reporting context.

Notaria_client_data/
Sample or historical client documents used for testing and dataset building.

Outputs generated per run

notary_summary.txt
Short summary of the run, including missing required documents and per-file status.

notary_detailed_report.txt
Detailed per-file reasoning including validation issues and legal basis labels.

notary_file_report.csv
Spreadsheet-friendly per-file report including classification and matching signals.

notary_phase_outputs.txt
Full run output across the workflow steps in one text file.


7. What is working well right now

Document upload and folder scanning.
Text extraction for common formats and OCR fallback (when enabled).
Field extraction using regex and optional Groq extraction.
Content-based classification (keyword and optional Groq classification).
Rule-based validation for missing documents, missing required fields, expiry checks, and cross-document consistency.
Clear reporting to short summary, detailed report, and CSV.
Hard stop (final confirmation gate) that prevents certificate generation when required items are missing.


8. What is incomplete or not working yet

Automatic retrieval or updating of missing documents from official sources is not implemented. The project currently requires manual upload or manual verification when a required document is missing.
Web search fallback is a stub. Enabling it does not currently fetch real results.
Client catalog matching depends heavily on filenames. If users upload files with generic names, catalog matching will not attach helpful metadata unless you extend the matching logic.
Some “required element” checks are conservative and may produce warnings that need manual review, especially for things like destination entity, document source, signature presence, and other fields that are not always reliably extracted.


9. What needs to be done next (practical improvements)

Implement official-source integrations for the update step (BPS/DGI/registry sources) or formalize a fully manual update workflow in the UI where the notary can upload the missing documents directly against each missing requirement.
Add a clear manual review UI to resolve classification conflicts (LLM vs keywords) and to let the notary map an uploaded file to a required document type when the filename/content heuristics are wrong.
Improve normalization and consistency logic for names and RUT to reduce false inconsistencies.
Extend client catalog matching to support content-based matching, or enforce a naming convention at upload time (rename step) so catalog and dataset matching are reliable.


10. How to run the project (basic)

1) Create and activate a virtual environment and install dependencies from requirements.txt.
2) Add GROQ_API_KEY to .env if you want LLM extraction/classification.
3) Run the Streamlit app (for example streamlit run chatbot.py).

If you want legacy DOC extraction and OCR, install the required system tools (LibreOffice, OCR tools) and ensure they are available on PATH.

