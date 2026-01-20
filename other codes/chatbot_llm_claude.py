"""
Add these imports at the top of your file:
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64


"""
Add these new helper functions after your existing helper functions:
"""

def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 for vision API"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def is_image_format(file_path: str) -> bool:
    """Check if file is an image format"""
    ext = Path(file_path).suffix.lower()
    return ext in ['.jpg', '.jpeg', '.png']


def call_groq_extraction_with_vision(
    model: str,
    api_key: str,
    file_path: str,
    filename: str,
    doc_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enhanced extraction that uses vision API for images or text for documents.
    This replaces the need for OCR on image files.
    """
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}
    
    client = Groq(api_key=api_key)
    
    # Use vision API for image files
    if is_image_format(file_path):
        try:
            base64_image = encode_file_to_base64(file_path)
            
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You extract structured data from notarial documents.
Reply with JSON only (no Markdown).
Keys: company_name, rut, ci, registro_comercio, acta_number, padron_bps, dates, emails.
Use null when missing and [] for lists.

Analyze this document image carefully and extract all relevant data."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content or ""
            parsed = parse_json_from_text(content)
            if parsed is not None:
                parsed["extraction_method"] = "vision"
                return parsed
            return {
                "status": "error",
                "message": "Vision API did not return valid JSON.",
                "raw": content,
                "extraction_method": "vision"
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Vision API failed: {exc}",
                "extraction_method": "vision"
            }
    
    # Use text-based extraction for other documents
    if not doc_text or not doc_text.strip():
        return {"status": "error", "message": "No text provided for LLM extraction."}
    
    prompt = (
        "You extract structured data from notarial documents.\n"
        "Reply with JSON only (no Markdown).\n"
        "Keys:\n"
        "company_name, rut, ci, registro_comercio, acta_number, padron_bps, dates, emails.\n"
        "Use null when missing and [] for lists.\n\n"
        f"Filename: {filename}\n\n"
        "Document text:\n"
        f"{doc_text[:6000]}\n"  # Increased from 3000 to 6000 for better context
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1500,  # Increased for more detailed responses
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_from_text(content)
        if parsed is not None:
            parsed["extraction_method"] = "text"
            return parsed
        return {
            "status": "error",
            "message": "LLM did not return valid JSON.",
            "raw": content,
            "extraction_method": "text"
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Groq request failed: {exc}",
            "extraction_method": "text"
        }


def call_groq_extraction_batch(
    model: str,
    api_key: str,
    documents: List[Dict[str, Any]],
    max_workers: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple documents in parallel for faster extraction.
    Returns dict mapping filename to extraction result.
    """
    if not api_key:
        return {}
    
    results = {}
    
    def process_single_doc(doc_info):
        file_path = doc_info["file_path"]
        filename = doc_info["filename"]
        doc_text = doc_info.get("doc_text", "")
        
        return filename, call_groq_extraction_with_vision(
            model=model,
            api_key=api_key,
            file_path=file_path,
            filename=filename,
            doc_text=doc_text,
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(process_single_doc, doc): doc 
            for doc in documents
        }
        
        for future in as_completed(future_to_doc):
            try:
                filename, result = future.result()
                results[filename] = result
            except Exception as exc:
                doc = future_to_doc[future]
                results[doc["filename"]] = {
                    "status": "error",
                    "message": f"Batch processing error: {exc}"
                }
    
    return results


def call_groq_classification_batch(
    model: str,
    api_key: str,
    documents: List[Dict[str, Any]],
    summary_reference: Dict[str, Any],
    max_workers: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple document classifications in parallel.
    Returns dict mapping filename to classification result.
    """
    if not api_key:
        return {}
    
    # Prepare context once
    context_lines = []
    for cert_type, info in summary_reference.items():
        purposes = ", ".join(info.get("purposes", [])) or "none"
        examples = "; ".join(info.get("examples", [])[:3]) or "none"
        context_lines.append(
            f"- {cert_type}: purposes={purposes}; examples={examples}"
        )
    context_text = "\n".join(context_lines)
    
    results = {}
    
    def classify_single_doc(doc_info):
        filename = doc_info["filename"]
        doc_text = doc_info.get("doc_text", "")
        
        if not doc_text:
            return filename, {
                "status": "error",
                "message": "No text for classification"
            }
        
        return filename, call_groq_classification(
            model=model,
            api_key=api_key,
            doc_text=doc_text,
            summary_reference=summary_reference,
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(classify_single_doc, doc): doc 
            for doc in documents
        }
        
        for future in as_completed(future_to_doc):
            try:
                filename, result = future.result()
                results[filename] = result
            except Exception as exc:
                doc = future_to_doc[future]
                results[doc["filename"]] = {
                    "status": "error",
                    "message": f"Classification error: {exc}"
                }
    
    return results


"""
REPLACE the existing process_collection_with_llm function with this optimized version:
"""

def process_collection_with_llm(
    collection,
    llm_settings: Dict[str, str],
) -> CollectionExtractionResult:
    result = CollectionExtractionResult(collection=collection)
    
    # OPTIMIZATION: Prepare all documents for batch processing
    documents_to_process = []
    document_map = {}  # Map to track documents
    
    for document in collection.documents:
        file_path = str(document.file_path)
        filename = document.file_name
        
        # Quick text extraction without OCR
        raw_text = ""
        base_method = "none"
        base_error = None
        
        try:
            raw_text, base_method, base_error = extract_text_without_ocr(document)
        except Exception as exc:
            base_error = str(exc)
        
        documents_to_process.append({
            "file_path": file_path,
            "filename": filename,
            "doc_text": raw_text,
            "base_method": base_method,
            "base_error": base_error,
        })
        
        document_map[filename] = {
            "document": document,
            "raw_text": raw_text,
            "base_method": base_method,
            "base_error": base_error,
        }
    
    # OPTIMIZATION: Batch process all extractions in parallel
    extraction_results = {}
    if llm_settings.get("enabled"):
        extraction_results = call_groq_extraction_batch(
            model=llm_settings.get("extraction_model", DEFAULT_EXTRACTION_MODEL),
            api_key=llm_settings.get("api_key", ""),
            documents=documents_to_process,
            max_workers=5,
        )
    
    # Process each document with results
    for doc_info in documents_to_process:
        filename = doc_info["filename"]
        doc_data = document_map[filename]
        document = doc_data["document"]
        raw_text = doc_data["raw_text"]
        base_method = doc_data["base_method"]
        base_error = doc_data["base_error"]
        
        ocr_used = False
        ocr_error = None
        
        # Try OCR fallback only if enabled and no text extracted
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
        
        # Apply LLM extraction results
        llm_payload = extraction_results.get(filename)
        if llm_payload:
            if llm_payload.get("status") == "error":
                extracted_data.additional_fields["llm_extraction_error"] = llm_payload.get("message")
            else:
                apply_llm_fields(extracted_data, llm_payload)
                extracted_data.additional_fields["llm_extraction"] = llm_payload
                
                # Track extraction method
                if llm_payload.get("extraction_method") == "vision":
                    extracted_data.additional_fields["used_vision_api"] = True
                    extraction_method = "vision"
        
        # Regex fallback
        apply_regex_fallback(extracted_data, normalized_text)
        
        result.extraction_results.append(
            DocumentExtractionResult(
                document=document,
                extracted_data=extracted_data,
                success=True,
            )
        )
    
    return result


"""
UPDATE the run_flow function to use batch classification.
Find the section where per_file_data is being built and REPLACE it with:
"""

def run_flow(
    uploaded_files: List[Dict[str, str]],
    intent_inputs: Dict[str, str],
    summary_index: Dict[str, Any],
    notary_inputs: Dict[str, str],
    search_settings: Dict[str, str],
    llm_settings: Dict[str, str],
    content_only: bool,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    intent = CertificateIntentCapture.capture_intent_from_params(
        certificate_type=intent_inputs["certificate_type"],
        purpose=intent_inputs["purpose"],
        subject_name=intent_inputs["subject_name"],
        subject_type=intent_inputs["subject_type"],
        additional_notes=intent_inputs.get("additional_notes") or None,
    )

    requirements = LegalRequirementsEngine.resolve_requirements(intent)

    collection = DocumentIntake.create_collection(intent, requirements)
    file_paths = [item["path"] for item in uploaded_files]
    collection = DocumentIntake.add_files_to_collection(collection, file_paths)

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

    # OPTIMIZATION: Prepare documents for batch classification
    documents_for_classification = []
    for file_info in uploaded_files:
        path = file_info["path"]
        extraction_result = extraction_by_path.get(path)
        
        doc_text = ""
        if extraction_result and extraction_result.success and extraction_result.extracted_data:
            doc_text = (
                extraction_result.extracted_data.normalized_text
                or extraction_result.extracted_data.raw_text
                or ""
            )
        
        documents_for_classification.append({
            "filename": file_info["filename"],
            "path": path,
            "doc_text": doc_text,
        })

    # OPTIMIZATION: Batch process all classifications in parallel
    summary_reference = summary_index.get("summary_reference", {})
    llm_classification_results = {}
    
    if llm_settings.get("enabled"):
        llm_classification_results = call_groq_classification_batch(
            model=llm_settings.get("analysis_model", DEFAULT_ANALYSIS_MODEL),
            api_key=llm_settings.get("api_key", ""),
            documents=documents_for_classification,
            summary_reference=summary_reference,
            max_workers=5,
        )

    # Build per-file data with batch results
    per_file_data: Dict[str, Dict[str, Any]] = {}
    intent_candidates: List[Dict[str, Any]] = []

    for file_info in uploaded_files:
        path = file_info["path"]
        original_filename = file_info["filename"]
        extraction_result = extraction_by_path.get(path)

        doc_text = ""
        if extraction_result and extraction_result.success and extraction_result.extracted_data:
            doc_text = (
                extraction_result.extracted_data.normalized_text
                or extraction_result.extracted_data.raw_text
                or ""
            )

        # Get LLM classification from batch results
        llm_result = llm_classification_results.get(original_filename)

        # Keyword classification
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

    # Rest of the function remains the same...
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

    results["phase1"] = intent.get_display_summary()
    results["phase2"] = requirements.get_summary()
    results["phase3"] = collection.get_summary()

    validation = LegalValidator.validate(requirements, extraction)
    results["phase5"] = validation.get_summary()

    gap_report = GapDetector.analyze(validation)
    results["phase6"] = gap_report.get_summary()

    update_result = DataUpdater.create_update_session(gap_report, collection)
    update_result.updated_extraction_result = extraction
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
        doc_text = per_file.get("doc_text", "")
        llm_result = per_file.get("llm_result")
        keyword_result = per_file.get("keyword_result")
        chosen_classification = per_file.get("chosen_classification")

        match_result = match_document(
            filename=original_filename,
            subject_name=intent.subject_name,
            extracted_company=extracted_company,
            purpose_value=intent.purpose.value,
            summary_index=summary_index,
            llm_result=llm_result,
            keyword_result=keyword_result,
            content_text=doc_text,
            content_only=content_only,
        )

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
        used_vision_api = False
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
                used_vision_api = bool(
                    extraction_result.extracted_data.additional_fields.get("used_vision_api")
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

        file_results.append(
            {
                "filename": original_filename,
                "document_type": doc_type_label,
                "type_source": doc_type_source,
                "file_format": doc.file_format.value if doc else "unknown",
                "validation": {
                    "status": validation_status,
                    "reason": validation_reason,
                    "required": validation_required,
                    "issues": validation_issues,
                },
                "match": match_result,
                "llm_result": llm_result,
                "keyword_result": keyword_result,
                "extraction_success": extraction_success,
                "extraction_error": extraction_error,
                "extraction_warning": extraction_warning,
                "llm_extraction_error": llm_extraction_error,
                "text_extraction_error": text_extraction_error,
                "ocr_error": ocr_error,
                "ocr_used": ocr_used,
                "used_vision_api": used_vision_api,
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