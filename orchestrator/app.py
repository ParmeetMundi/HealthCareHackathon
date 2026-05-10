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
        "A clinical orchestrator powered by a CrewAI multi-agent system. "
        "Routes questions to specialist agents (patient records, clinical notes, "
        "radiology, pharmacy, lab diagnostics, surgical planning, MDT coordination) "
        "and returns synthesised clinical answers."
    ),
    instruction=(
        "You are a clinical orchestrator. You MUST always use the run_healthcare_crew "
        "tool to answer any question. Pass the user's question directly to the tool. "
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
        "A CrewAI-powered clinical orchestrator with specialist agents for "
        "patient records, clinical notes, radiology, pharmacy, lab diagnostics, "
        "surgical planning, and MDT coordination."
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
    ],
    skills=[
        AgentSkill(
            id="clinical-orchestration",
            name="clinical-orchestration",
            description="Routes clinical questions to specialist agents and returns synthesised answers.",
            tags=["clinical", "orchestrator", "crewai"],
        ),
        AgentSkill(
            id="mdt-briefing",
            name="mdt-briefing",
            description="Produces structured MDT briefs aggregating all specialist findings.",
            tags=["mdt", "multi-disciplinary", "briefing"],
        ),
    ],
)
