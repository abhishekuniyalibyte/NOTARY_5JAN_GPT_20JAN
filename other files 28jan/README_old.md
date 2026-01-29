# AI-Powered Uruguayan Notarial Certificate Automation System

> [!NOTE]  
> test document path = `abhishekchoudhary@iByte:~/Downloads/notaria_5jan_chatgpt/Notaria_client_data/Azili SA$`

## 📋 Overview
This system is a **legal validation engine** that automates notarial certificates in Uruguay, validating documents against notarial law (Articles 248-255) and generating legally compliant outputs.

### Key Capabilities
- ✅ **Legal Intelligence:** Articles 248-255 and cross-references.
- ✅ **Automated Validation:** Document presence, validity, and consistency checks.
- ✅ **Smart Extraction:** OCR/Text extraction for RUT, CI, names, and dates.
- ✅ **Gap Detection:** Identifies missing or expired info with actionable fixes.
- ✅ **Human-in-the-Loop:** Learning module for notary feedback and edits.

---

## 🏗️ System Architecture (11 Phases)
The system follows a modular 11-phase workflow. Refer to [workflow.md](workflow.md) for details.

| Phase | Name | Description |
|:---|:---|:---|
| 1 | **Intent** | Captures certificate type, purpose, and subject. |
| 2 | **Requirements** | Resolves legal rules based on Uruguayan law. |
| 3 | **Intake** | Collects and indexes folders/files. |
| 4 | **Extraction** | OCR & text processing for structured data. |
| 5 | **Validation** | Checks compliance and cross-document consistency. |
| 6 | **Gap Detection** | Detects missing items and prioritizes fixes. |
| 7 | **Data Update** | Allows manual updates to resolve gaps. |
| 8 | **Confirmation** | Final 8-point legal compliance check. |
| 9 | **Generation** | Template-based certificate text generation. |
| 10 | **Review** | Human-in-the-loop audit and feedback loop. |
| 11 | **Output** | Final export to PDF, HTML, TXT, or JSON. |

---

## 🚀 Quick Start

### Installation
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Running & Testing
```bash
# Run the Streamlit Chatbot (Phases 1-11 UI)
streamlit run chatbot_openai.py

# Run all 224 validation tests
python3 -m pytest tests/ -v

# Run individual phases (e.g., Phase 1)
python3 src/phase1_certificate_intent.py
```

---

## 🔍 Tools & Examples

<details>
<summary><b>Historical Certificate Analyzer (Preprocessing)</b></summary>

Builds a knowledge base from your historical data before running the main workflow.
```bash
export OPENAI_API_KEY="your-key-here"
python3 analyze_historical_certificates.py --use-llm
```
</details>

<details>
<summary><b>Complete End-to-End Code Example</b></summary>

```python
from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine
# ... (imports for other phases)

# 1. Start with intent
intent = CertificateIntentCapture.capture_intent_from_params(
    certificate_type="certificado_de_personeria",
    purpose="BPS",
    subject_name="GIRTEC S.A.",
    subject_type="company"
)

# 2. Complete flow follows from Phase 2 to Phase 11
# See workflow.md for full logic implementation
```
</details>

---

## 📚 Technical Reference

### Project Structure
- `src/`: Core implementation files for all 11 phases.
- `tests/`: 224 unit and integration tests.
- `Notaria_client_data/`: Sample client documents.
- `cetificate from dataset/`: Input JSONs for matching.

### Legal Framework
Implemented based on **Uruguayan Notarial Regulations**:
- **Art. 130:** Identification rules.
- **Art. 248-255:** Certificate requirements, signature certification, and format.

### Quality & Performance
- **LOC:** ~11,000+ lines of Python.
- **Tests:** 224 passing tests (100% coverage across core logic).
- **Formats:** PDF*, DOCX*, HTML, JSON, TXT.

---

## 👤 Support & License
- **Documentation:** See [workflow.md](workflow.md), [steps.md](steps.md), and [HISTORICAL_ANALYSIS_README.md](HISTORICAL_ANALYSIS_README.md).
- **License:** Proprietary/To be determined.
