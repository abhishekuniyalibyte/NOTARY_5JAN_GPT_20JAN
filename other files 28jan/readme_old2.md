# NOTARIA PROJECT DETAILED README

## What this project does

This project helps a notary upload a set of documents, extract the important information from them, validate the set against predefined legal and institutional requirements, and only generate a certificate draft when the requirements are satisfied. The goal is to make it clear what is missing, what is inconsistent, what is expired, and what needs manual verification before proceeding.

## Inputs (what the notary provides)

The process starts in the Streamlit app.

- The notary selects the certificate type and the purpose or destination (for example BPS). These two inputs decide what the system will require and validate.
- The notary can provide the subject name (company/person). If it is not provided, the system may infer it from extracted text, but that can create inconsistencies if the uploaded documents contain multiple name variations.
- The notary uploads multiple files, or provides a local folder path. If a folder is selected, every file inside the folder is treated as part of the same request, so mixing different companies in one folder will cause consistency warnings and may block generation.
- Optionally, the notary can select a client catalog (loaded from `cetificate from dataset/client_file_catalogs.json`, which is created from a client Excel file such as `ACRISOUND.xlsx`). This catalog does not define the legal requirements; it is used for matching hints, expected formats, and better reporting.
- Optionally, the notary can enable OCR fallback and LLM extraction/classification (Groq). OCR helps with scanned documents. The LLM helps extract structured fields and classify documents by content, but low-confidence results still require manual review.

## Workflow (how the system processes the request end to end)

### File handling and preparation

If the notary uploads files, the app saves each file into a temporary local directory so the rest of the code can work with stable file paths. The original filename is still kept for reporting and for any filename-based matching. If the notary provides a local folder path, the app scans the directory and collects all files inside it.

### Document intake and basic metadata

Each file is converted into an internal document object with metadata like file format, size, and processing status. The system tries to detect a probable document type from filename keywords (estatuto, acta, bps, dgi, poder, etc.). If detection is not possible from the filename, later content-based detection can still help when text is available.

### Optional client catalog matching (from the client Excel conversion)

If a client catalog is selected, the system tries to match the uploaded filename to the catalog entries. If a match is found, it attaches metadata such as description, expected extensions, and whether the match was exact or approximate. This is used to warn when the uploaded file does not match what the client catalog expects. If the user uploads files with generic names like `1.pdf`, catalog matching usually cannot work unless filenames are renamed or catalog matching is extended to use content.

### Text extraction and OCR

The system extracts text from each document. For PDFs/DOCX/TXT it tries direct extraction. For legacy DOC it may use LibreOffice conversion or other fallback methods. If no text is extracted and OCR fallback is enabled, it attempts OCR for scanned PDFs and images. After extraction, the system normalizes the text so later matching and field extraction are more stable (for example, it reduces differences caused by accents, casing, and punctuation).

### Structured data extraction

From the extracted text, the system extracts key fields such as company name, RUT, CI, registry references, and dates. Regex extraction is always available as a fallback. If Groq is enabled and an API key is present, the system also asks the LLM to extract fields and return them in a structured format; regex is still used to fill gaps or validate basic patterns.

### Understanding what each document is

The system tries to label each uploaded document based on its content. It uses a keyword classifier and can also use an LLM classifier if enabled. The classifier is constrained by the taxonomy built from `cetificate from dataset/certificate_summary.json` so it stays within known categories instead of inventing new ones. This content-based classification is especially useful when the files are named generically.

### Building the requirement checklist (the source of what is needed)

Before validation, the system builds a checklist for the selected certificate type and purpose. This checklist is defined in code in `src/phase2_legal_requirements.py`. It includes which document types are mandatory, which are optional, which have expiry windows (for example 30 days for some institutional requirements), and the legal basis label used for reporting (article/law reference).

### Validation (how compliance is decided)

The system validates the uploaded set using rule-based checks against the checklist.

- It checks required document presence: any mandatory document type that is not present is reported as missing and is treated as a blocking issue.
- It checks required fields: for each required document type, it verifies that the expected fields are found in the extracted data (for example, a CI should be found in an identity document, and a RUT should be found where required). Missing required fields are reported as issues and can block generation depending on severity.
- It checks expiry and freshness: when a requirement has an expiry window, the system extracts dates from the document and compares them to the allowed timeframe. Expired documents are blocking. If a date cannot be reliably confirmed, the system can request manual verification of validity.
- It checks data consistency across documents: company name and RUT should not conflict across the set. If multiple values are found, the system reports inconsistencies and asks for confirmation or correction.
- It checks extraction quality: unsupported formats, empty text, very low confidence, and OCR-only reads are flagged because they reduce reliability and may require manual review.

### Legal article references and the local articles folder

When the system reports issues, it includes the legal basis label from the rule mappings. It can also attach a short excerpt from local article texts stored in the `articles/` folder so the notary/client can see context. The enforcement is not done by parsing the article text; enforcement is driven by the explicit rule checks and mappings in code.

### Gaps, manual review, and final decision

All issues are converted into a clear list of gaps and action items (missing documents, inconsistencies, review-needed items). If external official sources are not configured, the system cannot fetch missing documents automatically, so missing items must be uploaded manually or verified by the notary. A final go/no-go gate blocks certificate generation if blocking issues remain. If everything passes, the system generates a certificate draft, runs a review step, and produces the final outputs.

## Outputs (what the notary gets at the end)

- `notary_summary.txt`: short overview of the run (subject, file count, document labels, whether attention is required, and which required documents are missing).
- `notary_detailed_report.txt`: detailed explanation per file (classification label, validation status, issues found, and legal basis labels).
- `notary_file_report.csv`: spreadsheet report of each uploaded file (classification signals, validation signals, and optional client catalog metadata when available).
- `notary_phase_outputs.txt`: combined run log showing the full workflow output across the main processing steps.

## Where the required documents rules come from

The mapping is defined in `src/phase2_legal_requirements.py`. This file contains the rules engine that takes certificate type and purpose and returns a structured list of document requirements. Each requirement indicates whether it is mandatory, whether it expires, how many days are allowed, and the legal basis label used in reporting.

## What the dataset JSON files are used for

### `cetificate from dataset/certificate_summary.json`

Used as a taxonomy/reference for classification and dataset-style matching. It helps label documents by type/purpose and provides known categories, attributes, and examples. It does not decide what is legally required for a specific request.

### `cetificate from dataset/customers_index.json`

Index of historical files grouped by customer. This is mainly a preprocessing artifact used to build `certificate_summary.json` and other derived summaries. The runtime workflow does not use `customers_index.json` directly.

### `cetificate from dataset/client_file_catalogs.json`

Client-specific catalog created from a client-provided Excel file (for example `ACRISOUND.xlsx`). At runtime it is used to match uploaded filenames to expected client documents so the system can attach descriptions and expected formats, and warn when the uploaded file does not match what the catalog expects.

## What is working well right now

- Document upload and folder scanning.
- Text extraction for common formats and OCR fallback (when enabled).
- Field extraction using regex and optional Groq extraction.
- Content-based classification (keyword and optional Groq classification).
- Rule-based validation for missing documents, missing required fields, expiry checks, and cross-document consistency.
- Clear reporting to short summary, detailed report, and CSV.
- Hard stop that prevents certificate generation when required items are missing.

## What is incomplete or not working yet

- Automatic retrieval or updating of missing documents from official sources is not implemented. The project currently requires manual upload or manual verification when a required document is missing.
- Web search fallback is a stub. Enabling it does not currently fetch real results.
- Client catalog matching depends heavily on filenames. If users upload files with generic names, catalog matching will not attach helpful metadata unless you extend the matching logic.
- Some documents do not extract cleanly (especially scanned/low-quality files), so certain fields like CI/RUT/company names/dates may be missed or extracted inconsistently and require manual review.
- Some required-element checks are conservative and may produce warnings that need manual review, especially for destination entity, document source, signature presence, and other fields that are not always reliably extracted.

## What needs to be done next

- Implement official-source integrations for the update step (BPS/DGI/registry sources) or formalize a manual update workflow in the UI where the notary can upload missing documents directly against each missing requirement.
- Add a manual review UI to resolve classification conflicts (LLM vs keywords) and to let the notary map an uploaded file to a required document type when the filename/content heuristics are wrong.
- Improve normalization and consistency logic for names and RUT to reduce false inconsistencies.
- Extend client catalog matching to support content-based matching, or enforce a naming convention at upload time so catalog and dataset matching are reliable.

## How to run the project (basic)

1. Create and activate a virtual environment and install dependencies from requirements.txt.
2. Add GROQ_API_KEY to .env if you want LLM extraction/classification.
3. Run the Streamlit app (for example `streamlit run chatbot.py`).

If you want legacy DOC extraction and OCR, install the required system tools (LibreOffice, OCR tools) and ensure they are available on PATH.
