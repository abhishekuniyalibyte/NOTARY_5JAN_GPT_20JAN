# Project Status and Client Call Notes

## What I am doing now
- Running the 11-phase pipeline end-to-end with LLM extraction and classification to identify document types, required elements, and legal compliance.
- Strengthening Phase 5 to output a clear "correct vs incorrect" signal and a detailed issue list (missing docs, missing fields, expired docs, format rules, article compliance).
- Using `certificate_summary.json` as a fast, offline taxonomy for classification and matching.

## What is working
- Multi-file ingestion and extraction (PDF, DOC/DOCX, JPG/PNG) with LLM-assisted extraction.
- Document type classification from content (LLM + keyword fallback).
- Phase 5 validation produces a clear status with legal article checks and required elements.
- Phase 6 gap report produces a structured fix list (missing/expired/incorrect).

## What is not working or not complete yet
- Phase 7 "fix" is a stub: no official APIs are integrated yet (BPS, DGI, Registro).
- Full correctness cannot be guaranteed without authoritative sources or manual review.
- Phase 3 document type detection is still heuristic-based (filename patterns); content-based detection is improving but not perfect.
- Expiry checks rely on date extraction; if dates are missing in text, the system flags warnings/errors.

## How verification works now (high level)
- Phase 3: intake and preliminary type detection.
- Phachatbot_llm.py is mostly an orchestrator/UI. It calls the phases and displays their summaries.
So if you change logic inside phases 3/5/6/7, you usually don’t need to change chatbot_llm.py, unless:

You add new outputs that you want to display (e.g., a new “fix plan” list).
You change return structures (e.g., new fields in summaries).
You want new inputs/controls in the UI (toggles, thresholds, API keys).
In short:

Phase logic changes → no UI change required (unless you want to surface them).
New features visible in UI → update chatbot_llm.py.
If you tell me what new outputs you plan to add, I can confirm whether chatbot_llm.py needs edits.

make changes in phases like you mention accoridng to project details. Creation of notarial certificates requires multiple processes depending on the certificate type, each with its own specific steps. The complexity varies according to the case, for example expired BPS certificates, among others.
Process
When document X is uploaded, the agent must be able to read it and interpret Uruguayan law to determine whether all legal requirements are met. Based on these requirements, it must verify the validity of the information and whether it is up to date.
A flow:
• The agent already has all the relevant information loaded into its knowledge base regarding how notaries must operate and how documents must be delivered. This includes the Organic Law and the Notarial Regulations. https://www.aeu.org.uy/aucdocumento.aspx?996,13853
• The notary specifies the certificate that needs to be created (its type and purpose, such as certifying signatures or legal situations).
• The notary uploads the available documentation needed to create the certificate, either via Drive integration or direct upload.
• The agent compares the uploaded documentation with its knowledge base to identify missing documents, outdated documents, or documents containing incorrect information.
• The notary then chooses between uploading the updated documents or instructing the agent to attempt to update them using APIs or public/private websites that may contain updated information.
• The agent attempts to update the information and informs whether the update was successful or not, marking in red the items that were found and those that were changed.
• The agent analyses all the information uploaded by the notary.
• Once all the required elements are validated and corrected, the agent generates the certificate using a predefined template provided by the notary.
• The notary confirms whether everything is correct so the agent can learn or adjust its output if the notary indicates that something is not correct (for example, wording or template structure). This may involve updating the template or improving its programming.
Example
If the notary wants to certify a signature, the agent must be able to reference Article 250, which refers to Article 130, and verify all required documents according to those articles, as well as everything contained in Article 255. All notarial certificates must comply with Articles 248 to 255, including cross-references.
It is important that the entire process can be done from a mobile phone (this will be another stage - frontend development)
Title examination
When a person wants to buy a property, the notary must verify that the seller is the legitimate owner and that the property has no encumbrances. This requires registry information, municipal records, and data from public agencies. This is based on notarial requirements outlined here: https://publicaciones.fder.edu.uy/index.php/me/article/view/128/121
'

￼
Conversation interrupted
wouldnt my file [certificate_summary.json](cetificate from dataset/certificate_summary.json) be any useful or do i need official sources/APIs will you use for “fixing” (BPS, DGI, Registro, etc.) for everyfile that user will upload

￼
certificate_summary.json is still useful, but it cannot be the source of truth for “fixing.” It’s a local taxonomy + matching dataset. It helps you:

recognize document types,
match expected purposes,
provide examples and keywords.
For correctness or fixing, you need authoritative sources/APIs (BPS, DGI, Registro, etc.). Otherwise the bot can only guess or flag, not guarantee correctness.

So the right split is:

Use certificate_summary.json for classification and hints
Use official sources for verification and updates
If you want 100% accuracy, then yes—for every uploaded file that needs correction, you must verify against official sources (or have the user upload a corrected version).

If you want, I can:

Keep certificate_summary.json as your internal knowledge base
Add a “verification layer” that only activates when a doc is missing/expired/incorrect
Build hooks for external APIs (BPS/DGI/Registro) when you’re ready
then why do i even use this [certificate_summary.json](cetificate from dataset/certificate_summary.json) file. i can directly read user uploaded file and see what its type and then go mtchit with offical source?

￼
Use certificate_summary.json because it gives you fast, offline classification and matching without calling external sources every time.

If you skip it and only use official sources:

You’ll be slower and more expensive (API calls for every file).
You’ll still need a document type guess to know which registry/API to call.
Some sources won’t have all doc types, so you need a fallback taxonomy anyway.
Best practice is:

Step 1: Use certificate_summary.json + LLM to classify the document (cheap, local).
Step 2: Only if it’s required/critical or uncertain, call official sources to verify or fix.
So yes, you could skip it, but you’d lose speed, robustness, and a fallback. The summary file is your local map, the APIs are your truth check.

today i have call with client. what can i ask him? what can i tell him, if he ask how or what am i doing right now. how will i verify the user uploded document? how am i using [certificate_summary.json](cetificate from dataset/certificate_summary.json) in the task. what api/source i need from client side to verify document? answer all this question so i can have conversation with client

￼
Finished working
Here’s a compact talk‑track + questions for your client call.

What to tell the client (current approach)

se 4: extract text (and LLM/vision fallback in `chatbot_llm2.py`).
- Phase 5: validate required documents, required elements, expiry, format rules, and legal articles.
- Phase 6: generate a fix plan (what is missing/expired/incorrect).
- Phase 7: placeholder for updates (manual upload or public registry fetch).
- Phase 8+: final confirmation, generation, review, output.

## Why `certificate_summary.json` is still useful
- It is a local taxonomy of certificate types, purposes, attributes, and examples.
- It provides fast offline classification and matching without calling APIs.
- It helps decide which external source to call once the doc type is known.
- It is not the source of truth for correctness; it is a classification aid.

## What is required for "fixing" documents
- Official, authoritative sources for verification and updates:
  - BPS (social security status and padron)
  - DGI (tax compliance)
  - Registro de Comercio / Registro de la Propiedad
  - Any private/internal registries the notary already uses
- Without these, the system can only flag issues and request updated uploads.

## Next steps (engineering)
1) Add review queue output for "unknown/not_found" or low-confidence cases.
2) Integrate authoritative sources for Phase 7 (BPS/DGI/Registro).
3) Tighten Phase 3 doc type detection using content signals.
4) Expand document-specific field rules in Phase 5 based on legal requirements.
5) Add tests for expiry and format-rule checks.

## Client call talk-track
- "We classify documents locally using `certificate_summary.json` and LLMs, then validate against legal requirements. When a doc fails, we generate a fix plan. The actual fixing requires either updated uploads or verified data from official sources (BPS/DGI/Registro)."

## Questions to ask the client
1) Which official sources are approved and trusted for verification?
2) Do you already have API access/credentials for BPS, DGI, Registro?
3) What fields are mandatory per certificate type in your real workflow?
4) What is the definition of "correct" for each certificate type?
5) Do you want human approval before adding new rules or dataset entries?
6) Any constraints for data retention, audit trails, or security?

## Notes for the client
- Without authoritative sources, 100% correctness is not possible; we can only flag issues.
- The system is ready to ingest and classify; the "fix" step needs official data access.
