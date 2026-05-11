"""
orchestrator — CrewAI crew wiring.

Builds and returns a crewai.Crew configured as hierarchical with 8 specialist
agents. Agents are cached; tasks are selected dynamically per question based
on a lightweight keyword classifier so that simple queries skip irrelevant
specialist pipelines.

Usage:
    from orchestrator.agent import build_crew

    crew = build_crew("What are the patient's active medications?")
    result = crew.kickoff(inputs={"question": "What are the patient's active medications?"})
"""
import os
import logging
import re

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

_MAX_AGENT_ITER = int(os.getenv("MAX_AGENT_ITER", "4"))
_MAX_CREW_RPM = int(os.getenv("MAX_CREW_RPM", "30"))

# Per-agent iteration overrides — agents with simple fetch-and-report tasks
# need fewer iterations than the default. This prevents wasted LLM round-trips
# when data is sparse or empty.
_AGENT_MAX_ITER: dict[str, int] = {
    "orchestrator_agent": _MAX_AGENT_ITER,
    "patient_records_agent": _MAX_AGENT_ITER,  # needs up to 5 FHIR calls (demo, conditions, meds, allergies, care team)
    "clinical_notes_agent": 2,    # 2 calls: DocumentReference + DiagnosticReport
    "radiology_agent": 2,         # 2 calls: ImagingStudy + radiology reports; early exit if empty
    "pharmacist_agent": 2,        # 1 call to check meds, early exit if count=0
    "lab_diagnostics_agent": 2,   # 2 calls: observations + lab results
    "surgical_planning_agent": 2, # context-driven, at most 1 FHIR call for procedure history
    "mdt_coordination_agent": 1,  # pure synthesis from context, no tool calls needed
    "general_purpose_agent": 6,  # may need up to 4 FHIR calls + synthesis
}

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
        get_procedure_history,
        get_family_member_history,
    ],
    "mdt_coordination_agent": [],
    "general_purpose_agent": [
        get_patient_demographics,
        get_active_conditions,
        get_active_medications,
        get_allergies,
    ],
}

_RAG_TOOLS = [rag_store, rag_retrieve]

# ── Coworker descriptions for dynamic manager backstory ────────────────────────
# Maps agent keys to the coworker line the manager sees.
_COWORKER_DESCRIPTIONS: dict[str, str] = {
    "patient_records_agent": '"FHIR data retrieval specialist" — demographics, conditions, medications, allergies, care team',
    "clinical_notes_agent": '"Clinical document analyst" — clinical notes, discharge summaries',
    "radiology_agent": '"Radiology report interpreter" — imaging studies',
    "pharmacist_agent": '"Medication safety specialist" — drug interactions, dosage review',
    "lab_diagnostics_agent": '"Laboratory results interpreter" — lab results and vitals',
    "surgical_planning_agent": '"Pre/post-operative clinical assistant" — surgical risk assessment',
    "mdt_coordination_agent": '"Multi-disciplinary team meeting coordinator" — MDT synthesis',
    "general_purpose_agent": '"General healthcare assistant" — nutrition plans, exercise plans, lifestyle advice, and any other request not covered by the above specialists',
}

# ── Cached agents (built once, reused across requests) ─────────────────────────
_cached_agents: dict[str, Agent] | None = None
_cached_manager: Agent | None = None
_manager_backstory_template: str | None = None
_agents_cfg: dict | None = None
_tasks_cfg: dict | None = None


def _load_yaml(filename: str) -> dict:
    filepath = os.path.join(_CONFIG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Question classifier ───────────────────────────────────────────────────────
# Maps the incoming question to the minimal set of tasks needed.
# patient_records_task is ALWAYS included as the foundation.

_TASK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("clinical_notes_task", re.compile(
        r"note|document|discharge|summary|referral|progress|history\s+and\s+physical",
        re.IGNORECASE,
    )),
    ("radiology_task", re.compile(
        r"radiol|imaging|x[\-\s]?ray|ct\s+scan|mri|ultrasound|scan|cxr|chest",
        re.IGNORECASE,
    )),
    ("pharmacy_review_task", re.compile(
        r"medic|drug|pharma|interaction|dosage|prescription|contrain|pill|dosing",
        re.IGNORECASE,
    )),
    ("lab_diagnostics_task", re.compile(
        r"lab|blood\s*(test|work|count)?|vitals?|observation|cbc|hemoglobin|glucose|"
        r"creatinine|cholesterol|a1c|platelet|wbc|rbc|bmp|cmp|electrolyte|renal|liver\s+function",
        re.IGNORECASE,
    )),
    ("surgical_planning_task", re.compile(
        r"surg|operat|pre[\-\s]?op|post[\-\s]?op|anesthes|anaesthes|procedure\s+risk|asa\s+class",
        re.IGNORECASE,
    )),
]

# Patterns that trigger the full MDT pipeline (all tasks)
_MDT_PATTERN = re.compile(
    r"mdt|multi[\-\s]?disciplin|differential\s+diagnos|comprehensive|assessment|"
    r"treatment\s+(option|plan|recommend)|care\s+plan|full\s+review|"
    r"blood\s+in|bleeding|hematuria|present(ing|ation)|"
    r"what\s+is\s+wrong|diagnos|evaluate|assist",
    re.IGNORECASE,
)


def _classify_question(question: str) -> set[str]:
    """
    Return the set of task names needed for this question.

    Always includes patient_records_task. Adds mdt_synthesis_task only
    when 3+ specialist tasks are triggered or the question is explicitly
    comprehensive.
    """
    tasks = {"patient_records_task"}

    # If the question matches MDT-level complexity, run everything
    if _MDT_PATTERN.search(question):
        tasks.update(
            "clinical_notes_task", "radiology_task", "pharmacy_review_task",
            "lab_diagnostics_task", "surgical_planning_task", "mdt_synthesis_task",
        )
        logger.info("question_classified route=mdt_full question_len=%d", len(question))
        return tasks

    # Otherwise, match individual specialist tasks
    for task_name, pattern in _TASK_PATTERNS:
        if pattern.search(question):
            tasks.add(task_name)

    # Add surgical_planning_task dependencies if selected
    if "surgical_planning_task" in tasks:
        tasks.update(("lab_diagnostics_task", "pharmacy_review_task"))

    # Only synthesise if 3+ specialist tasks are running
    specialist_tasks = tasks - {"patient_records_task", "mdt_synthesis_task"}
    if len(specialist_tasks) >= 3:
        tasks.add("mdt_synthesis_task")

    # If no specialist matched beyond patient_records, route to the
    # general-purpose agent as a catch-all for nutrition plans, exercise
    # plans, lifestyle advice, and other non-specialist requests.
    if tasks == {"patient_records_task"}:
        tasks.add("general_purpose_task")

    # If only patient_records matched, that's fine — single-task crew
    route = "targeted" if len(tasks) <= 3 else "multi_specialist"
    logger.info(
        "question_classified route=%s tasks=%s question_len=%d",
        route, sorted(tasks), len(question),
    )
    return tasks


def _build_agents() -> tuple[dict[str, Agent], Agent]:
    """Build and cache all agents and the manager. Called once."""
    global _cached_agents, _cached_manager, _agents_cfg, _tasks_cfg

    if _cached_agents is not None and _cached_manager is not None:
        return _cached_agents, _cached_manager

    _agents_cfg = _load_yaml("agents.yaml")
    _tasks_cfg = _load_yaml("tasks.yaml")

    default_model = os.getenv("CREWAI_MODEL", "gpt-4o-mini")

    agents: dict[str, Agent] = {}
    for name, cfg in _agents_cfg.items():
        agent_tools = _AGENT_TOOLS.get(name, []) + _RAG_TOOLS
        agent_max_iter = _AGENT_MAX_ITER.get(name, _MAX_AGENT_ITER)
        agents[name] = Agent(
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg["backstory"],
            llm=LLM(model=default_model),
            tools=agent_tools,
            verbose=cfg.get("verbose", True),
            memory=cfg.get("memory", True),
            allow_delegation=cfg.get("allow_delegation", False),
            max_iter=agent_max_iter,
        )
        logger.info("agent_created name=%s model=%s tools=%d max_iter=%d", name, default_model, len(agent_tools), agent_max_iter)

    # Manager agent — delegation only, no FHIR tools
    model_name = os.getenv("CREWAI_MODEL", "gpt-4o-mini")
    manager_cfg = _agents_cfg.get("orchestrator_agent", {})
    manager = Agent(
        role=manager_cfg.get("role", "Clinical triage router and synthesiser"),
        goal=manager_cfg.get("goal", "Delegate to specialist agents and synthesise findings."),
        backstory=manager_cfg.get("backstory", "You are a senior clinical coordinator."),
        llm=LLM(model=model_name),
        tools=[],
        verbose=manager_cfg.get("verbose", True),
        memory=manager_cfg.get("memory", True),
        allow_delegation=True,
        max_iter=_MAX_AGENT_ITER,
    )
    logger.info("manager_agent_created model=%s tools=0 (delegation-only)", model_name)

    _cached_agents = agents
    _cached_manager = manager
    return agents, manager


def _build_tasks(
    tasks_cfg: dict,
    agents: dict[str, Agent],
    needed_tasks: set[str],
) -> dict[str, Task]:
    """Build and wire only the tasks in *needed_tasks*."""
    _ASYNC_TASKS = {"clinical_notes_task", "radiology_task", "pharmacy_review_task", "lab_diagnostics_task"}
    use_async = len(needed_tasks) > 2

    tasks: dict[str, Task] = {}
    for name, cfg in tasks_cfg.items():
        if name not in needed_tasks:
            continue
        agent_key = cfg["agent"]
        if agent_key not in agents:
            logger.warning("task_agent_not_found task=%s agent=%s", name, agent_key)
            continue
        tasks[name] = Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agents[agent_key],
            async_execution=name in _ASYNC_TASKS and use_async,
        )
        logger.info("task_created name=%s agent=%s async=%s", name, agent_key, name in _ASYNC_TASKS and use_async)

    # Wire context dependencies (only for tasks that exist in this run)
    for name, cfg in tasks_cfg.items():
        if "context" in cfg and name in tasks:
            context_tasks = [tasks[k] for k in cfg["context"] if k in tasks]
            if context_tasks:
                tasks[name].context = context_tasks
                logger.info("task_context_wired task=%s context=%s", name, [k for k in cfg["context"] if k in tasks])

    # CrewAI requires at most one async task at the end of the list.
    # Force the last task to be synchronous to satisfy this constraint.
    if tasks:
        last_task = list(tasks.values())[-1]
        if last_task.async_execution:
            last_task.async_execution = False
            logger.info("last_task_forced_sync name=%s", list(tasks.keys())[-1])

    return tasks


def build_crew(question: str = "") -> Crew:
    """
    Build and return a CrewAI Crew configured with hierarchical process.

    The agents are cached across requests. The task set is selected
    dynamically based on the question — simple queries only run the
    relevant specialist tasks, avoiding the full MDT pipeline.

    Args:
        question: The clinical question. Used to classify which tasks to run.
    """
    global _manager_backstory_template

    agents, manager_agent = _build_agents()
    tasks_cfg = _tasks_cfg or _load_yaml("tasks.yaml")

    needed_tasks = _classify_question(question)
    tasks = _build_tasks(tasks_cfg, agents, needed_tasks)

    # Collect worker agents (only those needed by selected tasks)
    needed_agent_keys = {tasks_cfg[t]["agent"] for t in needed_tasks if t in tasks_cfg}
    worker_agents = [a for name, a in agents.items() if name != "orchestrator_agent" and name in needed_agent_keys]

    # ── Dynamic manager backstory: list ONLY available coworkers ────────────
    # Prevents the manager from trying to delegate to agents not in this crew.
    if _manager_backstory_template is None:
        _manager_backstory_template = manager_agent.backstory

    coworker_lines = [
        f"    - {_COWORKER_DESCRIPTIONS[k]}"
        for k in needed_agent_keys
        if k in _COWORKER_DESCRIPTIONS
    ]
    coworker_section = "\n".join(coworker_lines)
    manager_agent.backstory = re.sub(
        r"(── COWORKER NAMES[^─]*──\n).*?(\n\s*── HARD RULES)",
        rf"\1{coworker_section}\2",
        _manager_backstory_template,
        flags=re.DOTALL,
    )
    logger.info("manager_backstory_updated available_coworkers=%s", sorted(needed_agent_keys))

    crew = Crew(
        agents=worker_agents,
        tasks=list(tasks.values()),
        process=Process.hierarchical,
        manager_agent=manager_agent,
        verbose=True,
        max_rpm=_MAX_CREW_RPM,
    )

    logger.info(
        "crew_built agents=%d tasks=%d process=hierarchical selected=%s",
        len(worker_agents), len(tasks), sorted(needed_tasks),
    )
    return crew
