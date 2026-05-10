"""
Shared tools catalogue — re-exports all tool functions available in this library.

FHIR tools (fhir.py)
────────────────────
  get_patient_demographics   Patient name, DOB, gender, contacts
  get_active_medications     Active MedicationRequest resources
  get_active_conditions      Active Condition resources (problem list)
  get_recent_observations    Observation resources — vitals, labs, etc.
  get_allergies              Active AllergyIntolerance resources
  get_care_team              CareTeam resources
  get_document_references    DocumentReference resources
  get_diagnostic_reports     DiagnosticReport resources
  get_imaging_studies        ImagingStudy resources
  get_radiology_reports      Radiology-category DiagnosticReport resources
  get_lab_results            Laboratory Observation resources
  get_procedure_history      Procedure resources

Pharmacy tools (pharmacy.py)
────────────────────────────
  check_drug_interactions    Built-in drug interaction checker
  get_medication_info        Built-in medication reference
"""

from .fhir import (
    get_active_conditions,
    get_active_medications,
    get_allergies,
    get_care_team,
    get_diagnostic_reports,
    get_document_references,
    get_imaging_studies,
    get_lab_results,
    get_patient_demographics,
    get_procedure_history,
    get_radiology_reports,
    get_recent_observations,
    set_fhir_context,
)

from .pharmacy import (
    check_drug_interactions,
    get_medication_info,
)

__all__ = [
    # FHIR tools
    "get_patient_demographics",
    "get_active_medications",
    "get_active_conditions",
    "get_recent_observations",
    "get_allergies",
    "get_care_team",
    "get_document_references",
    "get_diagnostic_reports",
    "get_imaging_studies",
    "get_radiology_reports",
    "get_lab_results",
    "get_procedure_history",
    # FHIR context
    "set_fhir_context",
    # Pharmacy tools
    "check_drug_interactions",
    "get_medication_info",
]
