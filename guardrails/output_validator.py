"""
Output guardrails — validate CrewAI crew output before returning to the caller.

Checks for hallucination indicators, empty output, FHIR grounding,
PII/token leakage, and MDT structural completeness.
"""
import re
import logging

logger = logging.getLogger(__name__)

_HALLUCINATION_PHRASES = [
    "i assume",
    "probably",
    "i think",
    "it is likely that",
    "i cannot confirm",
    "based on my training",
    "i believe",
    "most likely",
    "i'm not sure",
    "i am not sure",
    "presumably",
    "in my opinion",
]

_CLINICAL_CLAIM_PATTERN = re.compile(
    r"(\d+\s*(mg|mcg|ml|mmol|g/dL|mg/dL|mEq|units|IU|%|mmHg))"
    r"|(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)"
    r"|(\b(?:aspirin|metformin|lisinopril|atorvastatin|amlodipine|omeprazole"
    r"|warfarin|insulin|heparin|clopidogrel|metoprolol|losartan|furosemide"
    r"|prednisone|amoxicillin|ciprofloxacin|hydrochlorothiazide|gabapentin"
    r"|sertraline|pantoprazole|levothyroxine)\b)",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+/=_\-]{41,}")

_MDT_REQUIRED_SECTIONS = [
    "Patient Background",
    "Radiology",
    "Lab Summary",
    "Medication Review",
    "Treatment Recommendations",
    "Surgical Considerations",
    "Recommendations",
]

_MDT_TRIGGER_PHRASES = [
    "mdt",
    "multi-disciplinary",
    "multidisciplinary",
    "team meeting",
    "mdt brief",
    "mdt summary",
]


def validate_crew_output(output: str, agent_role: str, question: str = "") -> dict:
    """
    Validate the crew's output for safety and quality.

    Args:
        output: The raw text output from the CrewAI crew.
        agent_role: The role of the agent (e.g. "orchestrator").
        question: The original user question (used for MDT check).

    Returns:
        dict with keys: valid (bool), warnings (list[str]),
        errors (list[str]), sanitized_output (str).
    """
    warnings: list[str] = []
    errors: list[str] = []
    sanitized = output or ""

    # 1. Empty output guard
    if not sanitized or len(sanitized.strip()) < 20 or not re.search(r"[a-zA-Z]", sanitized):
        errors.append("Output is empty or too short to be clinically useful.")
        return {
            "valid": False,
            "warnings": warnings,
            "errors": errors,
            "sanitized_output": sanitized,
        }

    # 2. Hallucination guard
    output_lower = sanitized.lower()
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in output_lower:
            if _CLINICAL_CLAIM_PATTERN.search(sanitized):
                warnings.append(
                    f"Potential hallucination detected: output contains uncertain language "
                    f"('{phrase}') alongside clinical claims. Verify data against FHIR source."
                )
                break

    # 3. FHIR grounding check
    has_number = bool(re.search(r"\d", sanitized))
    has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", sanitized))
    has_proper_noun = bool(re.search(r"[A-Z][a-z]{2,}", sanitized))
    if not (has_number or has_date or has_proper_noun):
        warnings.append(
            "Output lacks concrete data points (numbers, dates, or proper nouns). "
            "May not be grounded in FHIR data."
        )

    # 4. PII/token leakage check
    token_matches = _TOKEN_PATTERN.findall(sanitized)
    for token_match in token_matches:
        if not _is_safe_long_string(token_match):
            sanitized = sanitized.replace(token_match, "[REDACTED]")
            warnings.append(
                "Potential token or credential detected and redacted from output."
            )

    # 5. MDT structural check
    if agent_role == "orchestrator":
        question_lower = (question or "").lower()
        is_mdt_request = any(phrase in question_lower for phrase in _MDT_TRIGGER_PHRASES)
        if is_mdt_request:
            missing_sections = []
            for section in _MDT_REQUIRED_SECTIONS:
                if section.lower() not in sanitized.lower():
                    missing_sections.append(section)
            if missing_sections:
                warnings.append(
                    f"MDT brief is missing required sections: {', '.join(missing_sections)}. "
                    f"Expected all of: {', '.join(_MDT_REQUIRED_SECTIONS)}"
                )

    valid = len(errors) == 0
    if warnings:
        logger.warning("guardrail_warnings agent=%s warnings=%s", agent_role, warnings)
    if errors:
        logger.error("guardrail_errors agent=%s errors=%s", agent_role, errors)

    return {
        "valid": valid,
        "warnings": warnings,
        "errors": errors,
        "sanitized_output": sanitized,
    }


def _is_safe_long_string(s: str) -> bool:
    """Check if a long string is likely safe (e.g. a URL path, base64 medical data)."""
    if s.startswith("http") or s.startswith("https"):
        return True
    if "/" in s and "." in s:
        return True
    if s.count("=") > 2 and len(s) > 100:
        return False
    alnum_ratio = sum(c.isalnum() for c in s) / max(len(s), 1)
    if alnum_ratio > 0.95 and len(s) > 60:
        return False
    return True
