"""
orchestrator — CrewAI crew wiring.

Builds and returns a crewai.Crew configured as hierarchical with 8 specialist
agents. All agents run in-process — no sub-agent has its own server or port.

Usage:
    from orchestrator.agent import build_crew

    crew = build_crew()
    result = crew.kickoff(inputs={"question": "What are the patient's active medications?"})
"""
import os
import logging

import yaml
from crewai import Agent, Task, Crew, Process, LLM

from shared.tools import (
    get_patient_demographics,
    get_active_medications,
    get_active_conditions,
    get_recent_observations,
    get_allergies,
    get_care_team,
    get_document_references,
    get_diagnostic_reports,
    get_imaging_studies,
    get_radiology_reports,
    get_lab_results,
    get_procedure_history,
    check_drug_interactions,
    get_medication_info,
)
from rag.tool import rag_store, rag_retrieve

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

# ── Tool assignments per agent ─────────────────────────────────────────────────
# RAG tools (rag_store, rag_retrieve) are added to every agent.

_AGENT_TOOLS = {
    "orchestrator_agent": [],
    "patient_records_agent": [
        get_patient_demographics,
        get_active_conditions,
        get_active_medications,
        get_allergies,
        get_care_team,
    ],
    "clinical_notes_agent": [
        get_document_references,
        get_diagnostic_reports,
    ],
    "radiology_agent": [
        get_imaging_studies,
        get_radiology_reports,
    ],
    "pharmacist_agent": [
        get_active_medications,
        get_active_conditions,
        check_drug_interactions,
        get_medication_info,
    ],
    "lab_diagnostics_agent": [
        get_recent_observations,
        get_lab_results,
    ],
    "surgical_planning_agent": [
        get_active_conditions,
        get_active_medications,
        get_procedure_history,
    ],
    "mdt_coordination_agent": [],
}

_RAG_TOOLS = [rag_store, rag_retrieve]


def _load_yaml(filename: str) -> dict:
    filepath = os.path.join(_CONFIG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_crew() -> Crew:
    """
    Build and return a CrewAI Crew configured with hierarchical process.

    Loads agent and task definitions from config/agents.yaml and config/tasks.yaml,
    instantiates all 8 agents with their tools, wires task context dependencies,
    and returns a Crew ready for kickoff.
    """
    agents_cfg = _load_yaml("agents.yaml")
    tasks_cfg = _load_yaml("tasks.yaml")

    # ── Build agents ──────────────────────────────────────────────────────
    # Default model — used when agent's llm field is not set in YAML
    default_model = os.getenv("CREWAI_MODEL", "gpt-4o-mini")

    agents: dict[str, Agent] = {}
    for name, cfg in agents_cfg.items():
        agent_tools = _AGENT_TOOLS.get(name, []) + _RAG_TOOLS
        agent_model = cfg.get("llm", default_model)
        agents[name] = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=LLM(model=agent_model),
            tools=agent_tools,
            verbose=cfg.get("verbose", True),
            memory=cfg.get("memory", True),
            allow_delegation=cfg.get("allow_delegation", False),
        )
        logger.info("agent_created name=%s model=%s tools=%d", name, agent_model, len(agent_tools))

    # ── Build tasks (without context first) ───────────────────────────────
    tasks: dict[str, Task] = {}
    for name, cfg in tasks_cfg.items():
        agent_key = cfg["agent"]
        if agent_key not in agents:
            logger.warning("task_agent_not_found task=%s agent=%s", name, agent_key)
            continue
        tasks[name] = Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agents[agent_key],
        )
        logger.info("task_created name=%s agent=%s", name, agent_key)

    # ── Wire context dependencies ─────────────────────────────────────────
    for name, cfg in tasks_cfg.items():
        if "context" in cfg and name in tasks:
            context_tasks = []
            for ctx_key in cfg["context"]:
                if ctx_key in tasks:
                    context_tasks.append(tasks[ctx_key])
                else:
                    logger.warning("task_context_not_found task=%s context=%s", name, ctx_key)
            tasks[name].context = context_tasks
            logger.info("task_context_wired task=%s context=%s", name, cfg["context"])

    # ── Build crew ────────────────────────────────────────────────────────
    model_name = os.getenv("CREWAI_MODEL", "gpt-4o-mini")
    manager_llm = LLM(model=model_name)

    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=Process.hierarchical,
        manager_llm=manager_llm,
        verbose=True,
    )

    logger.info(
        "crew_built agents=%d tasks=%d process=hierarchical manager_model=%s",
        len(agents), len(tasks), model_name,
    )
    return crew
