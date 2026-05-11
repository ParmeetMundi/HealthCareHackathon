"""
orchestrator — A2A application entry point.

This is the ONLY A2A endpoint in the entire project. All specialist agents
run in-process inside the CrewAI hierarchical crew — no sub-agent has its
own server or port.

A thin ADK bridge agent wraps the CrewAI crew so we can reuse the existing
shared/app_factory.py factory without modification.

Start the server with:
    uvicorn orchestrator.app:a2a_app --host 0.0.0.0 --port 8003

The agent card is served publicly at:
    GET http://localhost:8003/.well-known/agent-card.json

All other endpoints require an X-API-Key header (see shared/middleware.py).
"""
import logging
import os

from a2a.types import AgentSkill
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

from shared.app_factory import create_a2a_app
from shared.db import init_db
from shared.fhir_hook import extract_fhir_context
from shared.tools.fhir import set_fhir_context
from shared.patient_chat_store import add_turn, get_context_for_prompt
from guardrails.output_validator import validate_crew_output
from .agent import build_crew

logger = logging.getLogger(__name__)

# Initialise PostgreSQL schema on import (tables created IF NOT EXISTS)
try:
    init_db()
    logger.info("postgres_schema_initialised")
except Exception as exc:
    logger.error("postgres_init_failed error=%s", exc)


# ── Bridge tool: runs the CrewAI crew ──────────────────────────────────────────

def run_healthcare_crew(question: str, tool_context: ToolContext) -> str:
    """
    Execute the CrewAI healthcare crew with the given clinical question.

    Bridges FHIR credentials from ADK session state to the module-level
    _fhir_ctx dict used by CrewAI tools, runs the crew, validates the
    output via guardrails, and returns the result.

    Args:
        question: The clinical question to answer. Pass the user's
                  question exactly as received.
    """
    # Bridge FHIR context from ADK session state → CrewAI module-level dict
    fhir_url = tool_context.state.get("fhir_url", "")
    fhir_token = tool_context.state.get("fhir_token", "")
    patient_id = tool_context.state.get("patient_id", "")

    if fhir_url and fhir_token and patient_id:
        set_fhir_context(fhir_url, fhir_token, patient_id)
        logger.info(
            "fhir_context_bridged patient_id=%s fhir_url=%s",
            patient_id, fhir_url,
        )

    # ── Retrieve patient chat history and prepend as context ───────────
    enriched_question = question
    if patient_id:
        try:
            chat_context = get_context_for_prompt(patient_id)
            if chat_context:
                enriched_question = chat_context + question
                logger.info(
                    "chat_history_injected patient_id=%s context_len=%d",
                    patient_id, len(chat_context),
                )
            # Store the user's question
            add_turn(patient_id, "user", question)
        except Exception as exc:
            logger.warning("chat_history_error action=retrieve error=%s", exc)

    # Build and run the CrewAI crew (tasks selected based on question)
    logger.info("crew_kickoff question_len=%d", len(enriched_question))
    crew = build_crew(enriched_question)
    result = crew.kickoff(inputs={"question": enriched_question})
    raw_output = str(result)

    # Validate output via guardrails
    validation = validate_crew_output(raw_output, "orchestrator", question)

    if not validation["valid"]:
        error_msg = "; ".join(validation["errors"])
        logger.error("crew_output_invalid errors=%s", error_msg)
        return f"The system could not produce a valid clinical response. Errors: {error_msg}"

    if validation["warnings"]:
        logger.warning("crew_output_warnings warnings=%s", validation["warnings"])

    # ── Store the assistant response in patient chat history ───────────
    if patient_id:
        try:
            add_turn(patient_id, "assistant", validation["sanitized_output"])
        except Exception as exc:
            logger.warning("chat_history_error action=store_response error=%s", exc)

    return validation["sanitized_output"]


# ── ADK bridge agent ───────────────────────────────────────────────────────────
# This thin ADK agent exists solely to satisfy create_a2a_app()'s interface.
# It delegates all work to the CrewAI crew via the run_healthcare_crew tool.

_model_name = os.getenv("CREWAI_MODEL", "gpt-4o-mini")
_model = LiteLlm(model=_model_name)

root_agent = Agent(
    name="orchestrator",
    model=_model,
    description=(
        "An AI-powered clinical decision support system that acts as a virtual "
        "attending physician. Queries live FHIR R4 patient data through specialist "
        "agents (patient records, clinical documents, radiology, pharmacy, lab "
        "diagnostics, surgical planning, MDT coordination), synthesises findings, "
        "and provides evidence-based clinical reasoning — including differential "
        "diagnoses, treatment recommendations, risk stratification, and care plans."
    ),
    instruction=(
        "You are a clinical decision support gateway. You have ONE tool: run_healthcare_crew. "
        "For EVERY user message, you MUST:\n"
        "1. Call run_healthcare_crew with the user's question passed VERBATIM as the 'question' argument.\n"
        "2. Return the tool's output EXACTLY as received — do NOT rewrite, summarise, or add commentary.\n"
        "3. If the output states data is unavailable (e.g. 'no imaging studies'), preserve that finding "
        "as the lead statement. Do NOT substitute unrelated data.\n"
        "4. NEVER answer clinical questions from your own knowledge — ALWAYS delegate to the crew.\n"
        "5. Do NOT call the tool more than once per user message.\n"
        "6. Do NOT add disclaimers, headers, or formatting not present in the crew output."
    ),
    tools=[run_healthcare_crew],
    before_model_callback=extract_fhir_context,
)


# ── A2A application ───────────────────────────────────────────────────────────

a2a_app = create_a2a_app(
    agent=root_agent,
    name="orchestrator",
    description=(
        "An AI clinical decision support system powered by a multi-agent CrewAI crew. "
        "Acts as a virtual attending physician — queries live FHIR R4 patient data, "
        "performs clinical reasoning, and returns evidence-based assessments covering "
        "diagnosis, treatment, medication safety, lab interpretation, surgical risk, "
        "and multi-disciplinary team coordination."
    ),
    url=os.getenv("ORCHESTRATOR_URL", os.getenv("BASE_URL", "http://localhost:8003")),
    port=8003,
    fhir_extension_uri=f"{os.getenv('PO_PLATFORM_BASE_URL', 'http://localhost:5139')}/schemas/a2a/v1/fhir-context",
    fhir_scopes=[
        {"name": "patient/Patient.rs",              "required": True},
        {"name": "patient/MedicationRequest.rs",     "required": True},
        {"name": "patient/Condition.rs",             "required": True},
        {"name": "patient/Observation.rs",           "required": True},
        {"name": "patient/AllergyIntolerance.rs",    "required": True},
        {"name": "patient/CareTeam.rs",              "required": True},
        {"name": "patient/DocumentReference.rs",     "required": True},
        {"name": "patient/DiagnosticReport.rs",      "required": True},
        {"name": "patient/ImagingStudy.rs",          "required": True},
        {"name": "patient/Procedure.rs",             "required": True},
        {"name": "patient/Immunization.rs",          "required": True},
        {"name": "patient/Encounter.rs",             "required": True},
        {"name": "patient/CarePlan.rs",              "required": True},
        {"name": "patient/FamilyMemberHistory.rs",   "required": True},
        {"name": "patient/Coverage.rs",              "required": True},
        {"name": "patient/Appointment.rs",           "required": True},
        {"name": "patient/ServiceRequest.rs",        "required": True},
    ],
    skills=[
        # ── Doctor / Clinical reasoning skills ────────────────────────────
        AgentSkill(
            id="clinical-assessment",
            name="Clinical Assessment & Differential Diagnosis",
            description=(
                "Acts as an attending physician — reviews patient history, conditions, "
                "labs, vitals, imaging, and medications to produce a clinical assessment "
                "with differential diagnoses ranked by likelihood, supporting evidence, "
                "and recommended workup to narrow the differential."
            ),
            tags=["diagnosis", "differential", "clinical-assessment", "doctor",
                  "clinical-reasoning", "workup", "history", "physical-exam"],
        ),
        AgentSkill(
            id="treatment-recommendations",
            name="Treatment Planning & Recommendations",
            description=(
                "Generates evidence-based treatment recommendations considering the "
                "patient's conditions, allergies, current medications, lab values, and "
                "comorbidities. Covers pharmacological and non-pharmacological options, "
                "lifestyle modifications, and follow-up timelines."
            ),
            tags=["treatment", "therapy", "recommendations", "management",
                  "follow-up", "care-plan", "guidelines", "evidence-based"],
        ),
        AgentSkill(
            id="risk-stratification",
            name="Clinical Risk Stratification",
            description=(
                "Stratifies patient risk using conditions, family history, labs, vitals, "
                "and social history. Applies clinical scoring systems (e.g., CHA₂DS₂-VASc, "
                "Wells, HEART, Framingham) and flags high-risk patients requiring urgent "
                "intervention or closer monitoring."
            ),
            tags=["risk", "scoring", "stratification", "prognosis", "mortality",
                  "acuity", "triage", "severity"],
        ),
        AgentSkill(
            id="clinical-qa",
            name="Clinical Question Answering",
            description=(
                "Answers open-ended clinical questions about a patient — 'Why is this "
                "patient on warfarin?', 'Is this patient safe for discharge?', 'What "
                "caused the elevated creatinine?' — by correlating data across all "
                "available FHIR resources and reasoning through the clinical picture."
            ),
            tags=["question", "clinical-query", "explain", "why", "what",
                  "patient-question", "reasoning"],
        ),
        # ── Data retrieval skills ─────────────────────────────────────────
        AgentSkill(
            id="patient-summary",
            name="Patient Summary & Demographics",
            description=(
                "Retrieves comprehensive patient information including demographics, "
                "active conditions, medications, allergies, immunisations, insurance "
                "coverage, care team, encounters, and appointments from the FHIR server."
            ),
            tags=["patient", "demographics", "conditions", "medications", "allergies",
                  "immunizations", "encounters", "coverage", "appointments", "care-team"],
        ),
        AgentSkill(
            id="clinical-documents",
            name="Clinical Document Analysis",
            description=(
                "Retrieves and summarises clinical documents — progress notes, discharge "
                "summaries, referral letters, and diagnostic reports. Decodes base64 "
                "attachments and extracts key findings, diagnoses, and recommendations."
            ),
            tags=["documents", "notes", "discharge-summary", "referral", "clinical-notes"],
        ),
        AgentSkill(
            id="radiology-interpretation",
            name="Radiology Report Interpretation",
            description=(
                "Retrieves and interprets radiology imaging studies (X-ray, CT, MRI, "
                "ultrasound) and radiology-category diagnostic reports. Presents findings, "
                "impressions, and follow-up recommendations in structured clinical format."
            ),
            tags=["radiology", "imaging", "xray", "ct", "mri", "ultrasound"],
        ),
        AgentSkill(
            id="medication-safety",
            name="Medication Safety Review",
            description=(
                "Reviews active medications for drug-drug interactions, dosage "
                "appropriateness, and condition-based contraindications. Flags concerns "
                "with severity ratings (major/moderate/minor) and provides recommendations."
            ),
            tags=["pharmacy", "medications", "drug-interactions", "dosage", "contraindications"],
        ),
        AgentSkill(
            id="lab-diagnostics",
            name="Lab Results & Vitals Interpretation",
            description=(
                "Retrieves laboratory results and vital signs, flags out-of-range or "
                "critical values, identifies trends (improving/stable/worsening), and "
                "correlates findings with patient conditions."
            ),
            tags=["labs", "laboratory", "vitals", "blood-work", "diagnostics", "trends"],
        ),
        AgentSkill(
            id="surgical-risk-assessment",
            name="Surgical Risk Assessment",
            description=(
                "Evaluates surgical fitness using conditions, medications, labs, procedures, "
                "family history, and care plans. Estimates ASA status, flags perioperative "
                "medication adjustments, and outlines post-operative monitoring plans."
            ),
            tags=["surgery", "pre-operative", "post-operative", "risk-assessment", "perioperative"],
        ),
        AgentSkill(
            id="mdt-briefing",
            name="MDT Brief Generation",
            description=(
                "Produces a structured Multi-Disciplinary Team brief aggregating all "
                "specialist findings into sections: Patient Background, Radiology, Labs, "
                "Medication Review, Surgical Considerations, Outstanding Questions, and "
                "MDT Recommendations."
            ),
            tags=["mdt", "multi-disciplinary", "briefing", "team-meeting", "synthesis"],
        ),
    ],
)
