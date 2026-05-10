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
    get_immunizations,
    get_encounters,
    get_care_plans,
    get_family_member_history,
    get_coverage,
    get_appointments,
    get_service_requests,
    check_drug_interactions,
    get_medication_info,
)
from rag.tool import rag_store, rag_retrieve

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

# ── Iteration / performance limits ─────────────────────────────────────────────
# Controls how many loops each agent may perform before being forced to return.
# Lower values reduce latency; raise only if agents are returning incomplete data.

_MAX_AGENT_ITER = int(os.getenv("MAX_AGENT_ITER", "5"))
_MAX_CREW_RPM = int(os.getenv("MAX_CREW_RPM", "30"))

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
        get_immunizations,
        get_encounters,
        get_coverage,
        get_appointments,
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
        get_encounters,
        get_family_member_history,
        get_care_plans,
        get_service_requests,
    ],
    "mdt_coordination_agent": [],
}

_RAG_TOOLS = [rag_store, rag_retrieve]

# ── Singleton crew cache ───────────────────────────────────────────────────────
_cached_crew: Crew | None = None


def _load_yaml(filename: str) -> dict:
    filepath = os.path.join(_CONFIG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_crew() -> Crew:
    """
    Build and return a CrewAI Crew configured with hierarchical process.

    The crew is built once and cached — subsequent calls return the same
    instance to avoid re-parsing YAML and re-instantiating agents on every request.
    """
    global _cached_crew
    if _cached_crew is not None:
        logger.info("crew_cache_hit")
        return _cached_crew

    agents_cfg = _load_yaml("agents.yaml")
    tasks_cfg = _load_yaml("tasks.yaml")

    # ── Build agents ──────────────────────────────────────────────────────
    # Single LLM model for all agents — configured via CREWAI_MODEL env var.
    default_model = os.getenv("CREWAI_MODEL", "gpt-4o-mini")

    agents: dict[str, Agent] = {}
    for name, cfg in agents_cfg.items():
        agent_tools = _AGENT_TOOLS.get(name, []) + _RAG_TOOLS
        agents[name] = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=LLM(model=default_model),
            tools=agent_tools,
            verbose=cfg.get("verbose", True),
            memory=cfg.get("memory", True),
            allow_delegation=cfg.get("allow_delegation", False),
            max_iter=_MAX_AGENT_ITER,
        )
        logger.info("agent_created name=%s model=%s tools=%d max_iter=%d", name, default_model, len(agent_tools), _MAX_AGENT_ITER)

    # ── Build tasks (without context first) ───────────────────────────────
    # Tasks that only depend on patient_records_task and not on each other
    # can run in parallel via async_execution=True.
    _ASYNC_TASKS = {"clinical_notes_task", "radiology_task", "pharmacy_review_task", "lab_diagnostics_task"}

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
            async_execution=name in _ASYNC_TASKS,
        )
        logger.info("task_created name=%s agent=%s async=%s", name, agent_key, name in _ASYNC_TASKS)

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

    # Explicit manager agent — restricted to delegation only (no FHIR tools).
    # This prevents the Crew Manager from calling FHIR tools directly and
    # forces it to route work through the appropriate specialist agents.
    manager_cfg = agents_cfg.get("orchestrator_agent", {})
    manager_agent = Agent(
        role=manager_cfg.get("role", "Clinical triage router and synthesiser"),
        goal=manager_cfg.get("goal", "Delegate to specialist agents and synthesise findings."),
        backstory=manager_cfg.get("backstory", "You are a senior clinical coordinator."),
        llm=manager_llm,
        tools=[],  # Only RAG tools — no FHIR tools
        verbose=manager_cfg.get("verbose", True),
        memory=manager_cfg.get("memory", True),
        allow_delegation=True,
        max_iter=_MAX_AGENT_ITER,
    )
    logger.info("manager_agent_created model=%s tools=0 (delegation-only)", model_name)

    # Exclude orchestrator_agent from the worker agent list to avoid duplication.
    worker_agents = [a for name, a in agents.items() if name != "orchestrator_agent"]

    crew = Crew(
        agents=worker_agents,
        tasks=list(tasks.values()),
        process=Process.hierarchical,
        manager_agent=manager_agent,
        verbose=True,
        max_rpm=_MAX_CREW_RPM,
    )

    logger.info(
        "crew_built agents=%d tasks=%d process=hierarchical manager_model=%s max_rpm=%d max_agent_iter=%d",
        len(agents), len(tasks), model_name, _MAX_CREW_RPM, _MAX_AGENT_ITER,
    )
    _cached_crew = crew
    return crew
