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
  get_immunizations          Immunization (vaccination) records
  get_encounters             Encounter (visit/hospitalization) history
  get_care_plans             Active CarePlan resources
  get_family_member_history  FamilyMemberHistory resources
  get_coverage               Coverage / insurance information
  get_appointments           Appointment resources
  get_service_requests       ServiceRequest (referrals, orders)

Pharmacy tools (pharmacy.py)
────────────────────────────
  check_drug_interactions    Built-in drug interaction checker
  get_medication_info        Built-in medication reference
"""

from .fhir import (
    get_active_conditions,
    get_active_medications,
    get_allergies,
    get_appointments,
    get_care_plans,
    get_care_team,
    get_coverage,
    get_diagnostic_reports,
    get_document_references,
    get_encounters,
    get_family_member_history,
    get_imaging_studies,
    get_immunizations,
    get_lab_results,
    get_patient_demographics,
    get_procedure_history,
    get_radiology_reports,
    get_recent_observations,
    get_service_requests,
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
    "get_immunizations",
    "get_encounters",
    "get_care_plans",
    "get_family_member_history",
    "get_coverage",
    "get_appointments",
    "get_service_requests",
    # FHIR context
    "set_fhir_context",
    # Pharmacy tools
    "check_drug_interactions",
    "get_medication_info",
]
