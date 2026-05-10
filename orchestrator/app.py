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
from shared.fhir_hook import extract_fhir_context
from shared.tools.fhir import set_fhir_context
from guardrails.output_validator import validate_crew_output
from .agent import build_crew

logger = logging.getLogger(__name__)


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

    # Build and run the CrewAI crew
    logger.info("crew_kickoff question_len=%d", len(question))
    crew = build_crew()
    result = crew.kickoff(inputs={"question": question})
    raw_output = str(result)

    # Validate output via guardrails
    validation = validate_crew_output(raw_output, "orchestrator", question)

    if not validation["valid"]:
        error_msg = "; ".join(validation["errors"])
        logger.error("crew_output_invalid errors=%s", error_msg)
        return f"The system could not produce a valid clinical response. Errors: {error_msg}"

    if validation["warnings"]:
        logger.warning("crew_output_warnings warnings=%s", validation["warnings"])

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
        "You are a clinical decision support system acting as a senior attending physician. "
        "You MUST always use the run_healthcare_crew tool to answer any question. "
        "Pass the user's question directly to the tool. "
        "Return the tool's output as your final answer without modification. "
        "Do not attempt to answer clinical questions yourself — always delegate to the crew."
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
