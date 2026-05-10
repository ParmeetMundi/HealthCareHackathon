
```
You are an expert Python architect specializing in multi-agent AI systems, healthcare interoperability (FHIR R4), CrewAI, and the A2A protocol.

---

## STEP 0 — INPUT VALIDATION (do this before writing any code)

Before generating anything, evaluate whether you can fulfill this request completely and correctly. Check all of the following:

1. Do you have sufficient knowledge of CrewAI (>=0.80.0) agents, tasks, crews, and YAML config format to produce runnable code?
2. Do you have sufficient knowledge of the A2A v1 protocol and the `a2a-sdk` (0.3.x) to wire the orchestrator correctly?
3. Do you have sufficient knowledge of FHIR R4 REST API patterns (Patient, MedicationRequest, Condition, Observation, DocumentReference, DiagnosticReport, ImagingStudy, Procedure, AllergyIntolerance, CareTeam) to implement all tools?
4. Do you have sufficient knowledge of ChromaDB (>=0.5.0) persistent collections and `sentence-transformers` to implement the RAG layer?
5. Is every file in the deliverables list something you can generate completely — no stubs, no placeholders, no `...`?

If ANY answer is NO or UNCERTAIN, respond with:
> "I cannot fully implement this request because: [specific gap]. I can implement [what I can do] but would need [what is missing]."

Do NOT proceed to code generation if you cannot fulfill the request completely. Do NOT hallucinate library APIs, method signatures, or FHIR resource structures you are not certain of.

If ALL answers are YES, proceed to Step 1.

---

## STEP 1 — GOAL

Build a production-ready **CrewAI** multi-agent healthcare system that:

1. Retrieves patient data from a **FHIR R4** server using the same credential flow as described in the reference ADK project below.
2. Exposes **only the Orchestrator** as a single **A2A v1** endpoint on port 8003 using the existing `shared/app_factory.py` factory — do not modify it.
3. All specialist sub-agents run **entirely in-process** inside the CrewAI hierarchical crew — they are plain `crewai.Agent` objects, not separate servers. No sub-agent has its own `app.py` or port.
4. Uses **CrewAI** (`crewai>=0.80.0`) with agents and tasks declared in **YAML files** (`config/agents.yaml`, `config/tasks.yaml`).
5. Gives every agent **long-term memory** via a **RAG tool** backed by ChromaDB (`chromadb>=0.5.0`) and `sentence-transformers` (`all-MiniLM-L6-v2`).

---

## STEP 2 — REFERENCE PROJECT PATTERNS TO PRESERVE EXACTLY

### FHIR credential flow
FHIR credentials (`fhir_url`, `fhir_token`, `patient_id`) travel in the A2A message `params.message.metadata` under the key matching `fhir_extension_uri`. They are extracted by the orchestrator's `app.py` before the crew starts, stored in a module-level `_fhir_ctx` dict, and read from that dict by all FHIR tool functions at call time. Credentials never appear in any LLM prompt.

### A2A exposure — orchestrator only
Only `orchestrator/app.py` calls `create_a2a_app()` from `shared/app_factory.py`. This is the single public endpoint. Do not create `app.py` files for any sub-agent. Do not modify `shared/app_factory.py`.

### Security
The orchestrator sets `require_api_key=True`. `shared/middleware.py` `ApiKeyMiddleware` is reused without changes.

### Shared tools
All FHIR HTTP calls live in `shared/tools/fhir.py`. Each tool is a plain Python function decorated with `@tool` (CrewAI style) that reads FHIR credentials from a module-level `_fhir_ctx` dict. Export all tools from `shared/tools/__init__.py`.

### FHIR context bridging for CrewAI
1. `shared/tools/fhir.py` has a module-level dict: `_fhir_ctx: dict = {}`
2. A function `set_fhir_context(fhir_url: str, fhir_token: str, patient_id: str)` populates it
3. All `@tool` functions read from `_fhir_ctx` (never from env vars or hardcoded values)
4. `orchestrator/app.py` extracts FHIR credentials from the incoming A2A request body using `shared/fhir_hook.py`'s `extract_fhir_from_payload` and calls `set_fhir_context()` before calling `crew.kickoff()`

---

## STEP 3 — AGENTS TO IMPLEMENT

All agents are `crewai.Agent` instances instantiated inside `orchestrator/agent.py`. None are separate servers.

### 1. Orchestrator Agent (manager)
- **Role**: Clinical triage router and synthesiser
- **Goal**: Receive the clinician's or patient's question, decompose it, delegate to the right specialist agents using CrewAI's hierarchical process, and return a single coherent answer
- **CrewAI process**: `Process.hierarchical` — this agent is the `manager_llm`
- **RAG collection**: `"orchestrator_decisions"`

### 2. Patient Records Agent
- **Role**: FHIR data retrieval specialist
- **Goal**: Answer questions about demographics, active conditions, active medications, allergies, care team
- **FHIR tools**: `get_patient_demographics`, `get_active_conditions`, `get_active_medications`, `get_allergies`, `get_care_team`
- **RAG collection**: `"patient_records"`

### 3. Clinical Notes & Documents Agent
- **Role**: Clinical document analyst
- **Goal**: Retrieve, decode (base64 if needed), parse, and summarise clinical notes, discharge summaries, referral letters, and progress notes from FHIR `DocumentReference` and `DiagnosticReport`
- **FHIR tools**: `get_document_references`, `get_diagnostic_reports`
- **RAG collection**: `"clinical_notes"`

### 4. Radiology Agent
- **Role**: Radiology report interpreter
- **Goal**: Retrieve and summarise radiology reports (X-ray, CT, MRI, CXR) — findings, impression, recommendations
- **FHIR tools**: `get_imaging_studies`, `get_radiology_reports`
- **RAG collection**: `"radiology"`

### 5. Pharmacist Agent
- **Role**: Medication safety specialist
- **Goal**: Review active medications, check drug–drug interactions (built-in table, no external API), verify dosages against conditions, flag concerns
- **FHIR tools**: `get_active_medications`, `get_active_conditions`
- **Non-FHIR tools**: `check_drug_interactions(drug_a, drug_b)` (built-in table of 30+ common pairs), `get_medication_info(drug_name)` (standard dosage range, side effects, drug class)
- **RAG collection**: `"pharmacy_reviews"`

### 6. Lab & Diagnostics Agent
- **Role**: Laboratory results interpreter
- **Goal**: Retrieve lab results and vitals, flag out-of-range values, provide structured summary with trend indicators
- **FHIR tools**: `get_recent_observations`, `get_lab_results`
- **RAG collection**: `"lab_results"`

### 7. Surgical Planning Agent
- **Role**: Pre/post-operative clinical assistant
- **Goal**: Using conditions, medications, labs, and procedure history — summarise surgical risk factors, flag contraindications, outline post-op monitoring requirements
- **FHIR tools**: `get_active_conditions`, `get_active_medications`, `get_procedure_history`
- **RAG collection**: `"surgical_cases"`

### 8. MDT Coordination Agent
- **Role**: Multi-disciplinary team meeting coordinator
- **Goal**: Aggregate structured summaries from all other agents (passed as task context — does NOT call FHIR directly) and produce a structured MDT brief with sections: Patient Background, Radiology Findings, Lab Summary, Medication Review, Surgical Considerations, Outstanding Questions, MDT Recommendations
- **RAG collection**: `"mdt_summaries"`

---

## STEP 4 — PROJECT STRUCTURE

```
crewai_healthcare/
├── config/
│   ├── agents.yaml          ← all 8 agent definitions
│   └── tasks.yaml           ← all task definitions with context dependencies
├── orchestrator/
│   ├── __init__.py
│   ├── agent.py             ← CrewAI Crew wiring: all 8 agents + tasks, hierarchical process
│   └── app.py               ← ONLY A2A endpoint; calls create_a2a_app(), extracts FHIR context, kicks off crew
├── shared/                  ← copy verbatim; only fhir.py and tools/__init__.py are extended
│   ├── __init__.py
│   ├── app_factory.py
│   ├── fhir_hook.py
│   ├── logging_utils.py
│   ├── middleware.py
│   └── tools/
│       ├── __init__.py      ← updated exports for new tools
│       └── fhir.py          ← original 4 tools preserved + 8 new tools + _fhir_ctx + set_fhir_context()
├── rag/
│   ├── __init__.py
│   ├── store.py             ← ChromaDB client + custom sentence-transformers embedding function
│   └── tool.py              ← rag_store() and rag_retrieve() @tool functions
├── guardrails/
│   ├── __init__.py
│   └── output_validator.py  ← validate_crew_output() — called in orchestrator/app.py after kickoff
├── Procfile                 ← single line: web: uvicorn orchestrator.app:a2a_app --host 0.0.0.0 --port 8003
├── docker-compose.yml       ← single service on port 8003
├── Dockerfile
├── requirements.txt
└── .env.example
```

No sub-agent packages. No sub-agent `app.py` files. No sub-agent ports.

---

## STEP 5 — config/agents.yaml

Use CrewAI's standard YAML format. Every agent entry must include: `role`, `goal`, `backstory`, `verbose: true`, `allow_delegation` (true only for orchestrator), `memory: true`. Write all 8 agents fully.

---

## STEP 6 — config/tasks.yaml

Every task must include: `description`, `expected_output`, `agent` (key matching agents.yaml). Tasks that depend on outputs from other tasks must include a `context` list (list of task keys). Write all tasks fully:
- One primary task per specialist agent (tasks 2–8)
- One final MDT synthesis task with `context` pointing to all 7 specialist tasks

---

## STEP 7 — RAG layer (rag/store.py and rag/tool.py)

`rag/store.py`:
- Manages a single persistent ChromaDB client pointed at `CHROMA_PERSIST_DIR` env var (default `./chroma_db`)
- Exposes `get_collection(name: str)` that creates or retrieves a named collection
- Implements a custom ChromaDB embedding function using `sentence-transformers` `all-MiniLM-L6-v2`

`rag/tool.py` — two `@tool` decorated CrewAI functions:

```python
@tool("Store information in long-term memory")
def rag_store(collection_name: str, document: str, metadata: str) -> str:
    """Chunks document into 512-token segments with 50-token overlap,
    generates embeddings, upserts into the named ChromaDB collection.
    metadata must be a JSON string. Returns confirmation."""

@tool("Retrieve relevant information from long-term memory")
def rag_retrieve(collection_name: str, query: str, n_results: int = 5) -> str:
    """Embeds query and returns top n_results most similar chunks from
    the named ChromaDB collection as a formatted numbered list."""
```

Both tools are added to every agent's tools list in `orchestrator/agent.py`.

---

## STEP 8 — New FHIR tools to add to shared/tools/fhir.py

Add the following at the top of `shared/tools/fhir.py` (after existing imports):

```python
from crewai.tools import tool as crewai_tool

_fhir_ctx: dict = {}

def set_fhir_context(fhir_url: str, fhir_token: str, patient_id: str) -> None:
    _fhir_ctx["fhir_url"] = fhir_url.rstrip("/")
    _fhir_ctx["fhir_token"] = fhir_token
    _fhir_ctx["patient_id"] = patient_id
```

Refactor existing tools (`get_patient_demographics`, `get_active_medications`, `get_active_conditions`, `get_recent_observations`) to read from `_fhir_ctx` instead of `tool_context.state`, and decorate with `@crewai_tool`.

Add these new tools following the same pattern:

- `get_allergies` — `AllergyIntolerance?patient={id}&clinical-status=active`
- `get_care_team` — `CareTeam?patient={id}`
- `get_document_references` — `DocumentReference?patient={id}&_sort=-date&_count=20`
- `get_diagnostic_reports` — `DiagnosticReport?patient={id}&_sort=-date&_count=20`
- `get_imaging_studies` — `ImagingStudy?patient={id}&_sort=-started&_count=10`
- `get_radiology_reports` — `DiagnosticReport?patient={id}&category=LP29684-5&_sort=-date&_count=10`
- `get_lab_results` — `Observation?patient={id}&category=laboratory&_sort=-date&_count=50`
- `get_procedure_history` — `Procedure?patient={id}&_sort=-performed-date&_count=20`

Export all tools (old and new) and `set_fhir_context` from `shared/tools/__init__.py`.

---

## STEP 9 — orchestrator/agent.py

```python
# Build and return a crewai.Crew configured as hierarchical.
# Steps:
# 1. Load config/agents.yaml and config/tasks.yaml using CrewAI's YAML loader.
# 2. Instantiate all 8 crewai.Agent objects from the YAML definitions,
#    attaching the correct tool lists to each (FHIR tools + rag_store + rag_retrieve).
# 3. Instantiate all crewai.Task objects from the YAML definitions,
#    wiring context dependencies between tasks.
# 4. Return a crewai.Crew(
#        agents=[all 8 agents],
#        tasks=[all tasks],
#        process=Process.hierarchical,
#        manager_llm=LiteLlm(model=os.getenv("CREWAI_MODEL", "gemini/gemini-2.5-flash")),
#        verbose=True,
#    )
# Expose a build_crew() function that returns this Crew instance.
```

---

## STEP 10 — orchestrator/app.py

```python
# This is the only A2A app in the entire project.
# Steps:
# 1. Import build_crew from orchestrator.agent.
# 2. Create a Starlette ASGI app that wraps the A2A handler.
# 3. On each incoming POST /:
#    a. Parse the A2A JSON-RPC body.
#    b. Extract FHIR credentials using shared.fhir_hook.extract_fhir_from_payload.
#    c. Call shared.tools.fhir.set_fhir_context(fhir_url, fhir_token, patient_id).
#    d. Extract the user's question from params.message.parts[0].text.
#    e. Call crew = build_crew(); result = crew.kickoff(inputs={"question": question}).
#    f. Call guardrails.output_validator.validate_crew_output(result, "orchestrator").
#    g. If validation errors exist, return a safe error response via A2A JSON-RPC.
#    h. Otherwise return result.raw as the A2A text response.
# 4. Call create_a2a_app() with require_api_key=True,
#    fhir_extension_uri from PO_PLATFORM_BASE_URL env var, port=8003.
```

---

## STEP 11 — Output guardrails (guardrails/output_validator.py)

Implement `validate_crew_output(output: str, agent_role: str) -> dict` that checks:

1. **Hallucination guard** — flag if output contains phrases like "I assume", "probably", "I think", "it is likely that", "I cannot confirm", "based on my training" combined with clinical claim language (drug names, dosage numbers, lab values).
2. **Empty output guard** — flag if output is blank, fewer than 20 characters, or only whitespace/punctuation.
3. **FHIR grounding check** — verify output contains at least one concrete data point (a number, a date, a proper noun) rather than generic filler.
4. **PII/token leakage check** — scan for raw token-like strings (>40 chars of mixed alphanumeric) that should never appear in LLM output; redact if found.
5. **MDT structural check** — if `agent_role == "orchestrator"` and the question appears to request an MDT summary, verify output contains all required section headers: "Patient Background", "Radiology", "Lab Summary", "Medication Review", "Surgical Considerations", "Recommendations".

Return:
```python
{
    "valid": bool,
    "warnings": list[str],
    "errors": list[str],
    "sanitized_output": str
}
```

---

## STEP 12 — Environment variables (.env.example)

```env
# Model (LiteLLM prefix format)
CREWAI_MODEL=gemini/gemini-2.5-flash

# Model API keys
GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Orchestrator public URL (placed in A2A agent card)
ORCHESTRATOR_URL=http://localhost:8003

# API key authentication
API_KEYS=
API_KEY_PRIMARY=
API_KEY_SECONDARY=

# Prompt Opinion platform
PO_PLATFORM_BASE_URL=http://localhost:5139

# RAG persistent storage
CHROMA_PERSIST_DIR=./chroma_db

# Logging
LOG_FULL_PAYLOAD=true
LOG_HOOK_RAW_OBJECTS=false
```

---

## STEP 13 — Procfile and docker-compose.yml

`Procfile`:
```
web: uvicorn orchestrator.app:a2a_app --host 0.0.0.0 --port 8003
```

`docker-compose.yml`: single service named `orchestrator` on port 8003, same `Dockerfile` as reference project, reads env from `.env`.

---

## STEP 14 — DELIVERABLES CHECKLIST

Generate every file completely — no stubs, no `# TODO`, no `...` placeholders.

- [ ] `config/agents.yaml` — all 8 agents
- [ ] `config/tasks.yaml` — all tasks with context dependencies
- [ ] `rag/store.py` — ChromaDB client + custom embedding function
- [ ] `rag/tool.py` — `rag_store` and `rag_retrieve` tools
- [ ] `guardrails/__init__.py` and `guardrails/output_validator.py`
- [ ] `shared/tools/fhir.py` — refactored to use `_fhir_ctx`, all 12 tools present
- [ ] `shared/tools/__init__.py` — updated exports including `set_fhir_context`
- [ ] `orchestrator/__init__.py`, `orchestrator/agent.py`, `orchestrator/app.py`
- [ ] `Procfile`, `docker-compose.yml`, `Dockerfile`, `.env.example`, `requirements.txt`

Do not create `app.py` or `__init__.py` for any sub-agent package. Do not modify any other file in `shared/`.

---

## STEP 15 — OUTPUT SELF-VALIDATION (fix before responding)

1. **Single port** — confirm only port 8003 appears in Procfile, docker-compose.yml, and orchestrator/app.py. No other port references.
2. **No sub-agent servers** — confirm there are no `app.py` files outside `orchestrator/`.
3. **Import consistency** — every imported symbol exists in stdlib, requirements.txt, or the project itself. No invented imports.
4. **YAML validity** — mentally parse agents.yaml and tasks.yaml; keys match CrewAI's expected schema.
5. **FHIR context flow** — `set_fhir_context()` is called in `orchestrator/app.py` before `crew.kickoff()`. No tool reads credentials from env vars.
6. **Guardrail integration** — `validate_crew_output()` is called after every `crew.kickoff()` in `orchestrator/app.py` and errors are handled.
7. **RAG tools on all agents** — `rag_store` and `rag_retrieve` appear in every agent's tools list.
8. **No hardcoded secrets** — all credentials and keys come from env vars.
9. **No hallucinated APIs** — you have not used any ChromaDB, CrewAI, or sentence-transformers method that does not exist in the versions specified.