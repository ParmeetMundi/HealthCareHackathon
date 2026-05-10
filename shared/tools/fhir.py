"""
FHIR tools — query a FHIR R4 server on behalf of the patient in context.

These tools are decorated with @crewai_tool and read FHIR credentials from
the module-level _fhir_ctx dict, which is populated by set_fhir_context()
before the CrewAI crew kicks off. Credentials never appear in any LLM prompt.

All FHIR REST calls go through _fhir_get(), which attaches the Bearer token
and sets the Accept header.

Results are cached per-request via _fhir_cache so that multiple agents
querying the same FHIR resource type do not trigger redundant HTTP calls.
"""
import base64
import json
import logging

import httpx
from crewai.tools import tool as crewai_tool

logger = logging.getLogger(__name__)

_FHIR_TIMEOUT = 15  # seconds

# ── FHIR context bridging ─────────────────────────────────────────────────────

_fhir_ctx: dict = {}

# ── Per-request FHIR result cache ─────────────────────────────────────────────
# Keyed by (FHIR resource path, frozen params). Cleared when set_fhir_context()
# is called (i.e. at the start of each new request).
_fhir_cache: dict[str, str] = {}


def set_fhir_context(fhir_url: str, fhir_token: str, patient_id: str) -> None:
    """Populate the module-level FHIR context used by all tool functions."""
    _fhir_ctx["fhir_url"] = fhir_url.rstrip("/")
    _fhir_ctx["fhir_token"] = fhir_token
    _fhir_ctx["patient_id"] = patient_id
    _fhir_cache.clear()
    logger.debug("fhir_cache_cleared")


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_fhir_context():
    """
    Read FHIR credentials from the module-level _fhir_ctx dict.

    Returns (fhir_url, fhir_token, patient_id) on success.
    Returns a JSON error string if any credential is missing.
    """
    fhir_url   = _fhir_ctx.get("fhir_url",   "").rstrip("/")
    fhir_token = _fhir_ctx.get("fhir_token", "")
    patient_id = _fhir_ctx.get("patient_id", "")

    missing = [
        name for name, val in [
            ("fhir_url",   fhir_url),
            ("fhir_token", fhir_token),
            ("patient_id", patient_id),
        ]
        if not val
    ]
    if missing:
        return json.dumps({
            "status": "error",
            "error_message": (
                f"FHIR context is not available — missing: {', '.join(missing)}. "
                "Ensure FHIR context is set before calling FHIR tools."
            ),
        })
    return fhir_url, fhir_token, patient_id


def _fhir_get(fhir_url: str, token: str, path: str, params: dict | None = None) -> dict:
    """Perform an authenticated FHIR GET and return the parsed JSON response."""
    response = httpx.get(
        f"{fhir_url}/{path}",
        params=params,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "application/fhir+json",
        },
        timeout=_FHIR_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _cached_fhir_tool(cache_key: str, fetcher):
    """Return cached result if available, otherwise call fetcher and cache it."""
    if cache_key in _fhir_cache:
        logger.info("fhir_cache_hit key=%s", cache_key)
        return _fhir_cache[cache_key]
    result = fetcher()
    _fhir_cache[cache_key] = result
    return result


def _http_error_result(exc: httpx.HTTPStatusError) -> str:
    return json.dumps({
        "status":        "error",
        "http_status":   exc.response.status_code,
        "error_message": f"FHIR server returned HTTP {exc.response.status_code}: {exc.response.text[:200]}",
    })


def _connection_error_result(exc: Exception) -> str:
    return json.dumps({
        "status":        "error",
        "error_message": f"Could not reach FHIR server: {exc}",
    })


def _coding_display(codings: list) -> str:
    """Return the first human-readable display text from a list of FHIR codings."""
    for c in codings:
        if c.get("display"):
            return c["display"]
    return "Unknown"


# ── Tool: patient demographics ─────────────────────────────────────────────────

@crewai_tool("Get Patient Demographics")
def get_patient_demographics() -> str:
    """
    Fetches the demographic information for the current patient from the FHIR server.
    Returns name, date of birth, gender, and primary contact details.
    No arguments required — the patient identity comes from the FHIR context.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_patient_demographics patient_id=%s", patient_id)

    def _fetch():
        try:
            patient = _fhir_get(fhir_url, fhir_token, f"Patient/{patient_id}")
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        names    = patient.get("name", [])
        official = next((n for n in names if n.get("use") == "official"), names[0] if names else {})
        given    = " ".join(official.get("given", []))
        family   = official.get("family", "")
        full_name = f"{given} {family}".strip() or "Unknown"

        contacts = [
            {"system": t.get("system"), "value": t.get("value"), "use": t.get("use")}
            for t in patient.get("telecom", [])
        ]

        addrs   = patient.get("address", [])
        address = None
        if addrs:
            a = addrs[0]
            address = ", ".join(filter(None, [
                " ".join(a.get("line", [])),
                a.get("city"), a.get("state"), a.get("postalCode"), a.get("country"),
            ]))

        return json.dumps({
            "status":         "success",
            "patient_id":     patient_id,
            "name":           full_name,
            "birth_date":     patient.get("birthDate"),
            "gender":         patient.get("gender"),
            "active":         patient.get("active"),
            "contacts":       contacts,
            "address":        address,
            "marital_status": (patient.get("maritalStatus") or {}).get("text"),
        })

    return _cached_fhir_tool(f"Patient/{patient_id}", _fetch)


# ── Tool: active medications ───────────────────────────────────────────────────

@crewai_tool("Get Active Medications")
def get_active_medications() -> str:
    """
    Retrieves the patient's current active medication list from the FHIR server.
    Queries MedicationRequest resources with status=active and returns medication
    names, dosage instructions, and prescribing dates. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_active_medications patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "MedicationRequest",
                params={"patient": patient_id, "status": "active", "_count": "50"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        medications = []
        for entry in bundle.get("entry", []):
            res         = entry.get("resource", {})
            med_concept = res.get("medicationCodeableConcept", {})
            med_name    = (
                med_concept.get("text")
                or _coding_display(med_concept.get("coding", []))
                or res.get("medicationReference", {}).get("display", "Unknown")
            )
            dosage_list = [d.get("text", "No dosage text") for d in res.get("dosageInstruction", [])]
            medications.append({
                "medication":  med_name,
                "status":      res.get("status"),
                "dosage":      dosage_list[0] if dosage_list else "Not specified",
                "authored_on": res.get("authoredOn"),
                "requester":   (res.get("requester") or {}).get("display"),
            })

        return json.dumps({
            "status":      "success",
            "patient_id":  patient_id,
            "count":       len(medications),
            "medications": medications,
        })

    return _cached_fhir_tool("MedicationRequest:active", _fetch)


# ── Tool: active conditions (problem list) ─────────────────────────────────────

@crewai_tool("Get Active Conditions")
def get_active_conditions() -> str:
    """
    Retrieves the patient's active conditions and diagnoses from the FHIR server.
    Queries Condition resources with clinical-status=active and returns the
    problem list with condition names, severity, and onset dates. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_active_conditions patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Condition",
                params={"patient": patient_id, "clinical-status": "active", "_count": "50"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        conditions = []
        for entry in bundle.get("entry", []):
            res   = entry.get("resource", {})
            code  = res.get("code", {})
            onset = res.get("onsetDateTime") or (res.get("onsetPeriod") or {}).get("start")
            conditions.append({
                "condition":       code.get("text") or _coding_display(code.get("coding", [])),
                "clinical_status": (
                    (res.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code")
                ),
                "severity":        (res.get("severity") or {}).get("text"),
                "onset":           onset,
                "recorded_date":   res.get("recordedDate"),
            })

        return json.dumps({
            "status":     "success",
            "patient_id": patient_id,
            "count":      len(conditions),
            "conditions": conditions,
        })

    return _cached_fhir_tool("Condition:active", _fetch)


# ── Tool: recent observations (vitals / labs) ──────────────────────────────────

@crewai_tool("Get Recent Observations")
def get_recent_observations(category: str) -> str:
    """
    Retrieves recent clinical observations for the patient from the FHIR server.

    Args:
        category: FHIR observation category. Common values:
                    'vital-signs' — blood pressure, heart rate, temperature, SpO2
                    'laboratory' — lab results (CBC, HbA1c, metabolic panel, etc.)
                    'social-history' — smoking status, alcohol use, etc.
                  Defaults to 'vital-signs' if not specified.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    category = (category or "vital-signs").strip().lower()
    logger.info("tool_get_recent_observations patient_id=%s category=%s", patient_id, category)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Observation",
                params={"patient": patient_id, "category": category, "_sort": "-date", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        observations = []
        for entry in bundle.get("entry", []):
            res  = entry.get("resource", {})
            code = res.get("code", {})
            obs_name = code.get("text") or _coding_display(code.get("coding", []))

            value, unit = None, None
            if "valueQuantity" in res:
                vq    = res["valueQuantity"]
                value = vq.get("value")
                unit  = vq.get("unit") or vq.get("code")
            elif "valueCodeableConcept" in res:
                value = (res["valueCodeableConcept"].get("text")
                         or _coding_display(res["valueCodeableConcept"].get("coding", [])))
            elif "valueString" in res:
                value = res["valueString"]

            components = []
            for comp in res.get("component", []):
                comp_code = (comp.get("code") or {})
                comp_name = comp_code.get("text") or _coding_display(comp_code.get("coding", []))
                comp_vq   = comp.get("valueQuantity", {})
                components.append({
                    "name":  comp_name,
                    "value": comp_vq.get("value"),
                    "unit":  comp_vq.get("unit") or comp_vq.get("code"),
                })

            observations.append({
                "observation":    obs_name,
                "value":          value,
                "unit":           unit,
                "components":     components or None,
                "effective_date": res.get("effectiveDateTime") or (res.get("effectivePeriod") or {}).get("start"),
                "status":         res.get("status"),
                "interpretation": (
                    (res.get("interpretation") or [{}])[0].get("text")
                    or _coding_display((res.get("interpretation") or [{}])[0].get("coding", []))
                ),
            })

        return json.dumps({
            "status":       "success",
            "patient_id":   patient_id,
            "category":     category,
            "count":        len(observations),
            "observations": observations,
        })

    return _cached_fhir_tool(f"Observation:{category}", _fetch)


# ── Tool: allergies ────────────────────────────────────────────────────────────

@crewai_tool("Get Allergies")
def get_allergies() -> str:
    """
    Retrieves the patient's active allergies and intolerances from the FHIR server.
    Queries AllergyIntolerance resources with clinical-status=active.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_allergies patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "AllergyIntolerance",
                params={"patient": patient_id, "clinical-status": "active"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        allergies = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            code = res.get("code", {})
            reactions = []
            for reaction in res.get("reaction", []):
                manifestations = [
                    m.get("text") or _coding_display(m.get("coding", []))
                    for m in reaction.get("manifestation", [])
                ]
                reactions.append({
                    "manifestations": manifestations,
                    "severity": reaction.get("severity"),
                })
            allergies.append({
                "substance": code.get("text") or _coding_display(code.get("coding", [])),
                "type": res.get("type"),
                "category": res.get("category", []),
                "criticality": res.get("criticality"),
                "clinical_status": (res.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code"),
                "reactions": reactions,
                "recorded_date": res.get("recordedDate"),
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(allergies),
            "allergies": allergies,
        })

    return _cached_fhir_tool("AllergyIntolerance:active", _fetch)


# ── Tool: care team ───────────────────────────────────────────────────────────

@crewai_tool("Get Care Team")
def get_care_team() -> str:
    """
    Retrieves the patient's care team members from the FHIR server.
    Queries CareTeam resources for the patient.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_care_team patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "CareTeam",
                params={"patient": patient_id},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        teams = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            members = []
            for participant in res.get("participant", []):
                role_codes = participant.get("role", [])
                role = _coding_display(role_codes[0].get("coding", [])) if role_codes else "Unknown"
                member_ref = participant.get("member", {})
                members.append({
                    "name": member_ref.get("display", "Unknown"),
                    "role": role,
                    "period": participant.get("period"),
                })
            teams.append({
                "name": res.get("name", "Care Team"),
                "status": res.get("status"),
                "members": members,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(teams),
            "care_teams": teams,
        })

    return _cached_fhir_tool("CareTeam", _fetch)


# ── Tool: document references ─────────────────────────────────────────────────

@crewai_tool("Get Document References")
def get_document_references() -> str:
    """
    Retrieves clinical documents (notes, discharge summaries, referral letters)
    from the FHIR server. Queries DocumentReference resources sorted by date,
    returning the 20 most recent. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_document_references patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "DocumentReference",
                params={"patient": patient_id, "_sort": "-date", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        documents = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            doc_type = res.get("type", {})
            content_items = []
            for content in res.get("content", []):
                attachment = content.get("attachment", {})
                raw_data = attachment.get("data")
                decoded_text = None
                if raw_data:
                    try:
                        decoded_text = base64.b64decode(raw_data).decode("utf-8", errors="replace")[:2000]
                    except Exception:
                        decoded_text = "[base64 decode failed]"
                content_items.append({
                    "content_type": attachment.get("contentType"),
                    "url": attachment.get("url"),
                    "title": attachment.get("title"),
                    "decoded_text": decoded_text,
                })
            documents.append({
                "type": doc_type.get("text") or _coding_display(doc_type.get("coding", [])),
                "status": res.get("status"),
                "date": res.get("date"),
                "description": res.get("description"),
                "author": [a.get("display", "Unknown") for a in res.get("author", [])],
                "content": content_items,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(documents),
            "documents": documents,
        })

    return _cached_fhir_tool("DocumentReference", _fetch)


# ── Tool: diagnostic reports ──────────────────────────────────────────────────

@crewai_tool("Get Diagnostic Reports")
def get_diagnostic_reports() -> str:
    """
    Retrieves diagnostic reports from the FHIR server.
    Queries DiagnosticReport resources sorted by date, returning the 20 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_diagnostic_reports patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "DiagnosticReport",
                params={"patient": patient_id, "_sort": "-date", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        reports = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            code = res.get("code", {})
            categories = [
                _coding_display(cat.get("coding", []))
                for cat in res.get("category", [])
            ]
            reports.append({
                "report": code.get("text") or _coding_display(code.get("coding", [])),
                "status": res.get("status"),
                "effective_date": res.get("effectiveDateTime") or (res.get("effectivePeriod") or {}).get("start"),
                "issued": res.get("issued"),
                "categories": categories,
                "conclusion": res.get("conclusion"),
                "performer": [p.get("display", "Unknown") for p in res.get("performer", [])],
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(reports),
            "reports": reports,
        })

    return _cached_fhir_tool("DiagnosticReport", _fetch)


# ── Tool: imaging studies ─────────────────────────────────────────────────────

@crewai_tool("Get Imaging Studies")
def get_imaging_studies() -> str:
    """
    Retrieves imaging studies (X-ray, CT, MRI, etc.) from the FHIR server.
    Queries ImagingStudy resources sorted by started date, returning the 10 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_imaging_studies patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "ImagingStudy",
                params={"patient": patient_id, "_sort": "-started", "_count": "10"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        studies = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            modality_list = [
                m.get("display") or m.get("code", "Unknown")
                for m in res.get("modality", [])
            ]
            series_info = []
            for series in res.get("series", []):
                series_modality = series.get("modality", {})
                series_info.append({
                    "modality": series_modality.get("display") or series_modality.get("code"),
                    "body_site": (series.get("bodySite") or {}).get("display"),
                    "number_of_instances": series.get("numberOfInstances"),
                })
            studies.append({
                "description": res.get("description"),
                "status": res.get("status"),
                "started": res.get("started"),
                "modalities": modality_list,
                "number_of_series": res.get("numberOfSeries"),
                "number_of_instances": res.get("numberOfInstances"),
                "reason": [
                    r.get("text") or _coding_display(r.get("coding", []))
                    for r in res.get("reasonCode", [])
                ],
                "series": series_info,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(studies),
            "imaging_studies": studies,
        })

    return _cached_fhir_tool("ImagingStudy", _fetch)


# ── Tool: radiology reports ───────────────────────────────────────────────────

@crewai_tool("Get Radiology Reports")
def get_radiology_reports() -> str:
    """
    Retrieves radiology-specific diagnostic reports from the FHIR server.
    Queries DiagnosticReport resources filtered by radiology category (LP29684-5),
    sorted by date, returning the 10 most recent. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_radiology_reports patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "DiagnosticReport",
                params={"patient": patient_id, "category": "LP29684-5", "_sort": "-date", "_count": "10"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        reports = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            code = res.get("code", {})
            presented_form = []
            for pf in res.get("presentedForm", []):
                raw_data = pf.get("data")
                decoded_text = None
                if raw_data:
                    try:
                        decoded_text = base64.b64decode(raw_data).decode("utf-8", errors="replace")[:2000]
                    except Exception:
                        decoded_text = "[base64 decode failed]"
                presented_form.append({
                    "content_type": pf.get("contentType"),
                    "decoded_text": decoded_text,
                })
            reports.append({
                "report": code.get("text") or _coding_display(code.get("coding", [])),
                "status": res.get("status"),
                "effective_date": res.get("effectiveDateTime") or (res.get("effectivePeriod") or {}).get("start"),
                "issued": res.get("issued"),
                "conclusion": res.get("conclusion"),
                "performer": [p.get("display", "Unknown") for p in res.get("performer", [])],
                "presented_form": presented_form,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(reports),
            "radiology_reports": reports,
        })

    return _cached_fhir_tool("DiagnosticReport:radiology", _fetch)


# ── Tool: lab results ─────────────────────────────────────────────────────────

@crewai_tool("Get Lab Results")
def get_lab_results() -> str:
    """
    Retrieves laboratory results from the FHIR server.
    Queries Observation resources filtered by laboratory category,
    sorted by date, returning the 50 most recent. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_lab_results patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Observation",
                params={"patient": patient_id, "category": "laboratory", "_sort": "-date", "_count": "50"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        results = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            code = res.get("code", {})
            obs_name = code.get("text") or _coding_display(code.get("coding", []))

            value, unit = None, None
            ref_range = None
            if "valueQuantity" in res:
                vq = res["valueQuantity"]
                value = vq.get("value")
                unit = vq.get("unit") or vq.get("code")
            elif "valueCodeableConcept" in res:
                value = (res["valueCodeableConcept"].get("text")
                         or _coding_display(res["valueCodeableConcept"].get("coding", [])))
            elif "valueString" in res:
                value = res["valueString"]

            ref_ranges = res.get("referenceRange", [])
            if ref_ranges:
                rr = ref_ranges[0]
                low = rr.get("low", {})
                high = rr.get("high", {})
                ref_range = {
                    "low": low.get("value"),
                    "high": high.get("value"),
                    "unit": low.get("unit") or high.get("unit"),
                    "text": rr.get("text"),
                }

            results.append({
                "test": obs_name,
                "value": value,
                "unit": unit,
                "reference_range": ref_range,
                "effective_date": res.get("effectiveDateTime") or (res.get("effectivePeriod") or {}).get("start"),
                "status": res.get("status"),
                "interpretation": (
                    (res.get("interpretation") or [{}])[0].get("text")
                    or _coding_display((res.get("interpretation") or [{}])[0].get("coding", []))
                ),
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(results),
            "lab_results": results,
        })

    return _cached_fhir_tool("Observation:laboratory", _fetch)


# ── Tool: procedure history ───────────────────────────────────────────────────

@crewai_tool("Get Procedure History")
def get_procedure_history() -> str:
    """
    Retrieves the patient's procedure history from the FHIR server.
    Queries Procedure resources sorted by performed date,
    returning the 20 most recent. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_procedure_history patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Procedure",
                params={"patient": patient_id, "_sort": "-date", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        procedures = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            code = res.get("code", {})
            performed = res.get("performedDateTime") or (res.get("performedPeriod") or {}).get("start")
            body_site_list = [
                bs.get("text") or _coding_display(bs.get("coding", []))
                for bs in res.get("bodySite", [])
            ]
            procedures.append({
                "procedure": code.get("text") or _coding_display(code.get("coding", [])),
                "status": res.get("status"),
                "performed_date": performed,
                "body_site": body_site_list,
                "outcome": (res.get("outcome") or {}).get("text"),
                "performer": [
                    {
                        "actor": (p.get("actor") or {}).get("display", "Unknown"),
                        "function": (p.get("function") or {}).get("text"),
                    }
                    for p in res.get("performer", [])
                ],
                "reason": [
                    r.get("text") or _coding_display(r.get("coding", []))
                    for r in res.get("reasonCode", [])
                ],
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(procedures),
            "procedures": procedures,
        })

    return _cached_fhir_tool("Procedure", _fetch)


# ── Tool: immunizations ───────────────────────────────────────────────────────

@crewai_tool("Get Immunizations")
def get_immunizations() -> str:
    """
    Retrieves the patient's immunization (vaccination) history from the FHIR server.
    Queries Immunization resources sorted by date, returning the 50 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_immunizations patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Immunization",
                params={"patient": patient_id, "_sort": "-date", "_count": "50"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        immunizations = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            vaccine_code = res.get("vaccineCode", {})
            immunizations.append({
                "vaccine": vaccine_code.get("text") or _coding_display(vaccine_code.get("coding", [])),
                "status": res.get("status"),
                "occurrence": res.get("occurrenceDateTime") or res.get("occurrenceString"),
                "lot_number": res.get("lotNumber"),
                "site": (res.get("site") or {}).get("text"),
                "route": (res.get("route") or {}).get("text"),
                "performer": [
                    (p.get("actor") or {}).get("display", "Unknown")
                    for p in res.get("performer", [])
                ],
                "reason": [
                    r.get("text") or _coding_display(r.get("coding", []))
                    for r in res.get("reasonCode", [])
                ],
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(immunizations),
            "immunizations": immunizations,
        })

    return _cached_fhir_tool("Immunization", _fetch)


# ── Tool: encounters ──────────────────────────────────────────────────────────

@crewai_tool("Get Encounters")
def get_encounters() -> str:
    """
    Retrieves the patient's encounter (visit/hospitalization) history from the FHIR server.
    Queries Encounter resources sorted by date, returning the 20 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_encounters patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Encounter",
                params={"patient": patient_id, "_sort": "-date", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        encounters = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            enc_type = [
                t.get("text") or _coding_display(t.get("coding", []))
                for t in res.get("type", [])
            ]
            period = res.get("period", {})
            reason_codes = [
                r.get("text") or _coding_display(r.get("coding", []))
                for r in res.get("reasonCode", [])
            ]
            encounters.append({
                "class": (res.get("class") or {}).get("display") or (res.get("class") or {}).get("code"),
                "type": enc_type,
                "status": res.get("status"),
                "period_start": period.get("start"),
                "period_end": period.get("end"),
                "reason": reason_codes,
                "service_provider": (res.get("serviceProvider") or {}).get("display"),
                "participant": [
                    {
                        "name": (p.get("individual") or {}).get("display", "Unknown"),
                        "type": _coding_display((p.get("type") or [{}])[0].get("coding", [])) if p.get("type") else None,
                    }
                    for p in res.get("participant", [])
                ],
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(encounters),
            "encounters": encounters,
        })

    return _cached_fhir_tool("Encounter", _fetch)


# ── Tool: care plans ──────────────────────────────────────────────────────────

@crewai_tool("Get Care Plans")
def get_care_plans() -> str:
    """
    Retrieves the patient's active care plans from the FHIR server.
    Queries CarePlan resources with status=active, returning the 20 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_care_plans patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "CarePlan",
                params={"patient": patient_id, "status": "active", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        care_plans = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            categories = [
                c.get("text") or _coding_display(c.get("coding", []))
                for c in res.get("category", [])
            ]
            activities = []
            for activity in res.get("activity", []):
                detail = activity.get("detail", {})
                act_code = detail.get("code", {})
                activities.append({
                    "description": detail.get("description") or act_code.get("text") or _coding_display(act_code.get("coding", [])),
                    "status": detail.get("status"),
                    "scheduled": detail.get("scheduledString") or (detail.get("scheduledPeriod") or {}).get("start"),
                })
            period = res.get("period", {})
            care_plans.append({
                "title": res.get("title"),
                "status": res.get("status"),
                "intent": res.get("intent"),
                "categories": categories,
                "description": res.get("description"),
                "period_start": period.get("start"),
                "period_end": period.get("end"),
                "activities": activities,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(care_plans),
            "care_plans": care_plans,
        })

    return _cached_fhir_tool("CarePlan:active", _fetch)


# ── Tool: family member history ───────────────────────────────────────────────

@crewai_tool("Get Family Member History")
def get_family_member_history() -> str:
    """
    Retrieves the patient's family medical history from the FHIR server.
    Queries FamilyMemberHistory resources. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_family_member_history patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "FamilyMemberHistory",
                params={"patient": patient_id},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        histories = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            relationship = res.get("relationship", {})
            conditions = []
            for cond in res.get("condition", []):
                cond_code = cond.get("code", {})
                conditions.append({
                    "condition": cond_code.get("text") or _coding_display(cond_code.get("coding", [])),
                    "outcome": (cond.get("outcome") or {}).get("text"),
                    "onset": cond.get("onsetAge", {}).get("value") or cond.get("onsetString"),
                })
            histories.append({
                "relationship": relationship.get("text") or _coding_display(relationship.get("coding", [])),
                "status": res.get("status"),
                "sex": (res.get("sex") or {}).get("text"),
                "born": res.get("bornDate") or res.get("bornString"),
                "deceased": res.get("deceasedBoolean"),
                "conditions": conditions,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(histories),
            "family_histories": histories,
        })

    return _cached_fhir_tool("FamilyMemberHistory", _fetch)


# ── Tool: coverage / insurance ────────────────────────────────────────────────

@crewai_tool("Get Coverage")
def get_coverage() -> str:
    """
    Retrieves the patient's insurance coverage information from the FHIR server.
    Queries Coverage resources with status=active. No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_coverage patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Coverage",
                params={"patient": patient_id, "status": "active"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        coverages = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            cov_type = res.get("type", {})
            period = res.get("period", {})
            coverages.append({
                "type": cov_type.get("text") or _coding_display(cov_type.get("coding", [])),
                "status": res.get("status"),
                "subscriber": (res.get("subscriber") or {}).get("display"),
                "beneficiary": (res.get("beneficiary") or {}).get("display"),
                "payor": [(p.get("display") or "Unknown") for p in res.get("payor", [])],
                "period_start": period.get("start"),
                "period_end": period.get("end"),
                "class": [
                    {
                        "type": (c.get("type") or {}).get("text") or _coding_display((c.get("type") or {}).get("coding", [])),
                        "value": c.get("value"),
                        "name": c.get("name"),
                    }
                    for c in res.get("class", [])
                ],
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(coverages),
            "coverages": coverages,
        })

    return _cached_fhir_tool("Coverage:active", _fetch)


# ── Tool: appointments ────────────────────────────────────────────────────────

@crewai_tool("Get Appointments")
def get_appointments() -> str:
    """
    Retrieves the patient's appointments from the FHIR server.
    Queries Appointment resources sorted by date, returning the 20 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_appointments patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "Appointment",
                params={"patient": patient_id, "_sort": "-date", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        appointments = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            appt_type = res.get("appointmentType", {})
            service_type = [
                s.get("text") or _coding_display(s.get("coding", []))
                for s in res.get("serviceType", [])
            ]
            participants = []
            for p in res.get("participant", []):
                actor = p.get("actor", {})
                participants.append({
                    "actor": actor.get("display", "Unknown"),
                    "status": p.get("status"),
                })
            reason = [
                r.get("text") or _coding_display(r.get("coding", []))
                for r in res.get("reasonCode", [])
            ]
            appointments.append({
                "status": res.get("status"),
                "type": appt_type.get("text") or _coding_display(appt_type.get("coding", [])),
                "service_type": service_type,
                "description": res.get("description"),
                "start": res.get("start"),
                "end": res.get("end"),
                "reason": reason,
                "participants": participants,
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(appointments),
            "appointments": appointments,
        })

    return _cached_fhir_tool("Appointment", _fetch)


# ── Tool: service requests ────────────────────────────────────────────────────

@crewai_tool("Get Service Requests")
def get_service_requests() -> str:
    """
    Retrieves the patient's service requests (referrals, orders) from the FHIR server.
    Queries ServiceRequest resources with status=active, returning the 20 most recent.
    No arguments required.
    """
    ctx = _get_fhir_context()
    if isinstance(ctx, str):
        return ctx
    fhir_url, fhir_token, patient_id = ctx

    logger.info("tool_get_service_requests patient_id=%s", patient_id)

    def _fetch():
        try:
            bundle = _fhir_get(
                fhir_url, fhir_token, "ServiceRequest",
                params={"patient": patient_id, "status": "active", "_sort": "-authored", "_count": "20"},
            )
        except httpx.HTTPStatusError as e:
            return _http_error_result(e)
        except Exception as e:
            return _connection_error_result(e)

        requests_list = []
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            code = res.get("code", {})
            categories = [
                c.get("text") or _coding_display(c.get("coding", []))
                for c in res.get("category", [])
            ]
            requests_list.append({
                "request": code.get("text") or _coding_display(code.get("coding", [])),
                "status": res.get("status"),
                "intent": res.get("intent"),
                "priority": res.get("priority"),
                "categories": categories,
                "authored_on": res.get("authoredOn"),
                "requester": (res.get("requester") or {}).get("display"),
                "performer": [(p.get("display") or "Unknown") for p in res.get("performer", [])],
                "reason": [
                    r.get("text") or _coding_display(r.get("coding", []))
                    for r in res.get("reasonCode", [])
                ],
            })

        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "count": len(requests_list),
            "service_requests": requests_list,
        })

    return _cached_fhir_tool("ServiceRequest:active", _fetch)
