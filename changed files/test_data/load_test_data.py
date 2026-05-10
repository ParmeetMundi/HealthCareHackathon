"""
Load comprehensive FHIR R4 test patient data into a FHIR server.

Usage:
    python test_data/load_test_data.py --fhir-url http://localhost:8080/fhir

For a local HAPI FHIR server:
    docker run -p 8080:8080 hapiproject/hapi:latest
    python test_data/load_test_data.py

The script creates a realistic patient with:
  - Demographics, contacts, address
  - Conditions / diagnoses
  - Active medications
  - Allergies
  - Lab observations (CBC, metabolic panel, HbA1c, lipid panel)
  - Vital signs
  - DocumentReferences with embedded PDF & image attachments
  - DiagnosticReports with presentedForm PDFs (lab report, radiology)
  - ImagingStudy (chest X-ray)
  - Procedures
  - CareTeam
  - Encounters
"""

import argparse
import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import httpx

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FHIR_URL = "http://localhost:8080/fhir"
PATIENT_ID = "test-patient-001"
TIMEOUT = 30


# ── Tiny helper: generate a minimal valid PDF ─────────────────────────────────

def _make_pdf(title: str, body_lines: list[str]) -> bytes:
    """Generate a minimal but valid PDF document from plain-text lines."""
    body_text = "\\n".join(body_lines)
    stream = (
        f"BT /F1 14 Tf 50 750 Td ({title}) Tj ET\n"
        f"BT /F1 10 Tf 50 720 Td "
    )
    y = 720
    for line in body_lines:
        stream += f"({line}) Tj 0 -14 Td "
        y -= 14
    stream += "ET"

    objects = []
    # obj 1: catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    # obj 2: pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
    # obj 3: page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )
    # obj 4: content stream
    stream_bytes = stream.encode("latin-1")
    objects.append(
        b"4 0 obj\n<< /Length " + str(len(stream_bytes)).encode() + b" >>\nstream\n"
        + stream_bytes + b"\nendstream\nendobj"
    )
    # obj 5: font
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
    )

    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj + b"\n"

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += f"0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += b"trailer\n"
    pdf += f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    pdf += b"startxref\n"
    pdf += f"{xref_offset}\n".encode()
    pdf += b"%%EOF\n"
    return pdf


# ── Tiny helper: generate a minimal valid PNG (100x100 solid colour) ──────────

def _make_png(r: int = 200, g: int = 200, b: int = 200, w: int = 100, h: int = 100) -> bytes:
    """Generate a minimal valid PNG image (solid colour)."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = _chunk(b"IHDR", ihdr_data)

    raw_row = bytes([0] + [r, g, b] * w)  # filter byte 0 + RGB pixels
    raw_data = raw_row * h
    idat = _chunk(b"IDAT", zlib.compress(raw_data))
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


# ── Build the FHIR Transaction Bundle ─────────────────────────────────────────

def build_bundle() -> dict:
    entries = []

    def _add(resource: dict, resource_type: str | None = None) -> None:
        rt = resource_type or resource["resourceType"]
        rid = resource.get("id", "")
        entries.append({
            "resource": resource,
            "request": {
                "method": "PUT",
                "url": f"{rt}/{rid}" if rid else rt,
            },
        })

    # ── 1. Patient ─────────────────────────────────────────────────────────
    _add({
        "resourceType": "Patient",
        "id": PATIENT_ID,
        "active": True,
        "name": [
            {
                "use": "official",
                "family": "Martinez",
                "given": ["Elena", "Sofia"],
            }
        ],
        "gender": "female",
        "birthDate": "1978-03-15",
        "telecom": [
            {"system": "phone", "value": "+1-555-0142", "use": "home"},
            {"system": "email", "value": "elena.martinez@example.com", "use": "home"},
        ],
        "address": [
            {
                "use": "home",
                "line": ["742 Evergreen Terrace"],
                "city": "Springfield",
                "state": "IL",
                "postalCode": "62704",
                "country": "US",
            }
        ],
        "maritalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus", "code": "M", "display": "Married"}],
            "text": "Married",
        },
        "communication": [
            {"language": {"coding": [{"system": "urn:ietf:bcp:47", "code": "en", "display": "English"}]}, "preferred": True}
        ],
    })

    # ── 2. Conditions ──────────────────────────────────────────────────────
    conditions = [
        ("cond-001", "44054006", "Type 2 Diabetes Mellitus", "2015-06-20", "moderate"),
        ("cond-002", "38341003", "Essential Hypertension", "2016-01-10", "mild"),
        ("cond-003", "84114007", "Coronary Artery Disease", "2020-09-05", None),
        ("cond-004", "235595009", "Gastroesophageal Reflux Disease (GERD)", "2018-04-12", "mild"),
        ("cond-005", "396275006", "Osteoarthritis of Right Knee", "2022-11-30", "moderate"),
        ("cond-006", "267036007", "Iron Deficiency Anaemia", "2024-03-15", "mild"),
        ("cond-007", "190905008", "Hypothyroidism", "2019-07-22", None),
    ]
    for cid, snomed, display, onset, severity in conditions:
        cond = {
            "resourceType": "Condition",
            "id": cid,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "code": snomed, "display": display}],
                "text": display,
            },
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "onsetDateTime": onset,
            "recordedDate": onset,
        }
        if severity:
            cond["severity"] = {
                "coding": [{"system": "http://snomed.info/sct", "code": {"mild": "255604002", "moderate": "6736007"}[severity], "display": severity.capitalize()}],
                "text": severity.capitalize(),
            }
        _add(cond)

    # ── 3. Medications ─────────────────────────────────────────────────────
    meds = [
        ("med-001", "860975", "Metformin 1000mg", "1000mg twice daily with meals", "2015-06-25"),
        ("med-002", "314076", "Lisinopril 20mg", "20mg once daily in the morning", "2016-01-15"),
        ("med-003", "617311", "Atorvastatin 40mg", "40mg once daily at bedtime", "2020-09-10"),
        ("med-004", "198211", "Omeprazole 20mg", "20mg once daily before breakfast", "2018-04-15"),
        ("med-005", "310429", "Aspirin 81mg", "81mg once daily", "2020-09-10"),
        ("med-006", "197361", "Amlodipine 5mg", "5mg once daily", "2021-03-01"),
        ("med-007", "312961", "Levothyroxine 75mcg", "75mcg once daily on empty stomach", "2019-08-01"),
        ("med-008", "310261", "Ferrous Sulfate 325mg", "325mg once daily with vitamin C", "2024-03-20"),
    ]
    for mid, rxnorm, name, dosage, authored in meds:
        _add({
            "resourceType": "MedicationRequest",
            "id": mid,
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": rxnorm, "display": name}],
                "text": name,
            },
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "authoredOn": authored,
            "dosageInstruction": [{"text": dosage}],
            "requester": {"display": "Dr. James Wilson"},
        })

    # ── 4. Allergies ───────────────────────────────────────────────────────
    allergies = [
        ("allergy-001", "91936005", "Penicillin", "allergy", ["Urticaria", "Angioedema"], "high", "severe"),
        ("allergy-002", "387207008", "Ibuprofen", "intolerance", ["Dyspepsia", "Nausea"], "low", "mild"),
        ("allergy-003", "111088007", "Latex", "allergy", ["Contact dermatitis", "Pruritus"], "low", "moderate"),
    ]
    for aid, snomed, substance, atype, manifestations, criticality, severity in allergies:
        _add({
            "resourceType": "AllergyIntolerance",
            "id": aid,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
            "type": atype,
            "category": ["medication"] if atype == "allergy" and snomed != "111088007" else ["environment"],
            "criticality": criticality,
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "code": snomed, "display": substance}],
                "text": substance,
            },
            "patient": {"reference": f"Patient/{PATIENT_ID}"},
            "recordedDate": "2015-01-01",
            "reaction": [
                {
                    "manifestation": [{"coding": [{"display": m}], "text": m} for m in manifestations],
                    "severity": severity,
                }
            ],
        })

    # ── 5. Vital-sign observations ─────────────────────────────────────────
    vitals = [
        ("obs-vs-001", "85354-9", "Blood Pressure", None, None, "2025-04-15T09:30:00Z",
         [("8480-6", "Systolic", 138, "mmHg"), ("8462-4", "Diastolic", 82, "mmHg")]),
        ("obs-vs-002", "8867-4", "Heart Rate", 78, "beats/min", "2025-04-15T09:30:00Z", None),
        ("obs-vs-003", "8310-5", "Body Temperature", 36.8, "°C", "2025-04-15T09:30:00Z", None),
        ("obs-vs-004", "2708-6", "Oxygen Saturation", 97, "%", "2025-04-15T09:30:00Z", None),
        ("obs-vs-005", "29463-7", "Body Weight", 72.5, "kg", "2025-04-15T09:30:00Z", None),
        ("obs-vs-006", "8302-2", "Body Height", 165, "cm", "2025-04-15T09:30:00Z", None),
        ("obs-vs-007", "39156-5", "BMI", 26.6, "kg/m2", "2025-04-15T09:30:00Z", None),
    ]
    for oid, loinc, display, value, unit, effective, components in vitals:
        obs = {
            "resourceType": "Observation",
            "id": oid,
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}], "text": display},
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "effectiveDateTime": effective,
        }
        if value is not None:
            obs["valueQuantity"] = {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"}
        if components:
            obs["component"] = [
                {
                    "code": {"coding": [{"system": "http://loinc.org", "code": cl, "display": cd}], "text": cd},
                    "valueQuantity": {"value": cv, "unit": cu, "system": "http://unitsofmeasure.org"},
                }
                for cl, cd, cv, cu in components
            ]
        _add(obs)

    # ── 6. Lab observations ────────────────────────────────────────────────
    labs = [
        # CBC
        ("obs-lab-001", "718-7", "Hemoglobin", 11.2, "g/dL", "L", "12.0-16.0 g/dL"),
        ("obs-lab-002", "4544-3", "Hematocrit", 34.5, "%", "L", "36-46 %"),
        ("obs-lab-003", "6690-2", "WBC Count", 7.2, "10*3/uL", "N", "4.5-11.0 10*3/uL"),
        ("obs-lab-004", "777-3", "Platelet Count", 245, "10*3/uL", "N", "150-400 10*3/uL"),
        ("obs-lab-005", "789-8", "RBC Count", 3.9, "10*6/uL", "L", "4.0-5.5 10*6/uL"),
        ("obs-lab-006", "787-2", "MCV", 88.5, "fL", "N", "80-100 fL"),
        # Metabolic panel
        ("obs-lab-007", "2345-7", "Glucose (fasting)", 142, "mg/dL", "H", "70-100 mg/dL"),
        ("obs-lab-008", "4548-4", "HbA1c", 7.8, "%", "H", "< 5.7 %"),
        ("obs-lab-009", "2160-0", "Creatinine", 0.9, "mg/dL", "N", "0.6-1.2 mg/dL"),
        ("obs-lab-010", "3094-0", "BUN", 18, "mg/dL", "N", "7-20 mg/dL"),
        ("obs-lab-011", "2951-2", "Sodium", 140, "mEq/L", "N", "136-145 mEq/L"),
        ("obs-lab-012", "2823-3", "Potassium", 4.3, "mEq/L", "N", "3.5-5.0 mEq/L"),
        ("obs-lab-013", "33914-3", "eGFR", 78, "mL/min/1.73m2", "N", "> 60 mL/min/1.73m2"),
        # Lipid panel
        ("obs-lab-014", "2093-3", "Total Cholesterol", 215, "mg/dL", "H", "< 200 mg/dL"),
        ("obs-lab-015", "2571-8", "Triglycerides", 185, "mg/dL", "H", "< 150 mg/dL"),
        ("obs-lab-016", "2085-9", "HDL Cholesterol", 42, "mg/dL", "L", "> 40 mg/dL"),
        ("obs-lab-017", "13457-7", "LDL Cholesterol (calc)", 136, "mg/dL", "H", "< 100 mg/dL"),
        # Thyroid
        ("obs-lab-018", "3016-3", "TSH", 3.8, "mIU/L", "N", "0.4-4.0 mIU/L"),
        ("obs-lab-019", "3026-2", "Free T4", 1.1, "ng/dL", "N", "0.8-1.8 ng/dL"),
        # Iron studies
        ("obs-lab-020", "2498-4", "Serum Iron", 45, "mcg/dL", "L", "60-170 mcg/dL"),
        ("obs-lab-021", "2502-3", "Ferritin", 12, "ng/mL", "L", "12-150 ng/mL"),
        ("obs-lab-022", "2500-7", "TIBC", 420, "mcg/dL", "H", "250-370 mcg/dL"),
        # Liver function
        ("obs-lab-023", "1742-6", "ALT", 28, "U/L", "N", "7-56 U/L"),
        ("obs-lab-024", "1920-8", "AST", 25, "U/L", "N", "10-40 U/L"),
    ]
    for oid, loinc, display, value, unit, interp_code, ref_range in labs:
        interp_map = {"N": "Normal", "H": "High", "L": "Low"}
        _add({
            "resourceType": "Observation",
            "id": oid,
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory", "display": "Laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}], "text": display},
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "effectiveDateTime": "2025-04-10T08:00:00Z",
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"},
            "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": interp_code, "display": interp_map[interp_code]}], "text": interp_map[interp_code]}],
            "referenceRange": [{"text": ref_range}],
        })

    # ── 7. DocumentReferences with embedded PDF & image ────────────────────
    # 7a. Discharge summary PDF
    discharge_pdf = _make_pdf("Discharge Summary - Elena Martinez", [
        "Date: 2025-03-20",
        "MRN: TST-001",
        "",
        "Admission Date: 2025-03-15",
        "Discharge Date: 2025-03-20",
        "Attending: Dr. James Wilson",
        "",
        "Chief Complaint: Chest pain and shortness of breath",
        "",
        "Hospital Course:",
        "Patient admitted with acute chest pain. Cardiac catheterization",
        "revealed 70% stenosis of LAD. Managed medically with dual",
        "antiplatelet therapy and optimised statin dose.",
        "",
        "Discharge Medications:",
        "- Metformin 1000mg BID",
        "- Lisinopril 20mg daily",
        "- Atorvastatin 40mg daily (increased from 20mg)",
        "- Aspirin 81mg daily",
        "- Clopidogrel 75mg daily (new - 12 months)",
        "- Omeprazole 20mg daily",
        "- Amlodipine 5mg daily",
        "- Levothyroxine 75mcg daily",
        "",
        "Follow-up: Cardiology in 2 weeks, PCP in 1 week",
    ])
    _add({
        "resourceType": "DocumentReference",
        "id": "doc-001",
        "status": "current",
        "type": {
            "coding": [{"system": "http://loinc.org", "code": "18842-5", "display": "Discharge Summary"}],
            "text": "Discharge Summary",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "date": "2025-03-20T16:00:00Z",
        "author": [{"display": "Dr. James Wilson"}],
        "description": "Discharge summary following cardiac catheterisation admission March 2025",
        "content": [
            {
                "attachment": {
                    "contentType": "application/pdf",
                    "data": base64.b64encode(discharge_pdf).decode(),
                    "title": "Discharge_Summary_Martinez_20250320.pdf",
                }
            }
        ],
    })

    # 7b. Clinical note
    clinical_note_pdf = _make_pdf("Progress Note - Elena Martinez", [
        "Date: 2025-04-15",
        "Provider: Dr. James Wilson",
        "",
        "Subjective:",
        "Patient returns for follow-up. Reports good compliance with",
        "medications. Denies chest pain or shortness of breath since",
        "discharge. Mild right knee pain with activity.",
        "",
        "Objective:",
        "BP 138/82, HR 78, Temp 36.8C, SpO2 97%",
        "Heart: RRR, no murmurs. Lungs: clear bilaterally.",
        "Right knee: mild crepitus, no effusion.",
        "",
        "Assessment & Plan:",
        "1. CAD - stable post-cath, continue DAPT",
        "2. T2DM - HbA1c 7.8%, adjust metformin, recheck in 3 months",
        "3. HTN - borderline controlled, may increase amlodipine",
        "4. Knee OA - continue conservative management, PT referral",
        "5. Iron deficiency anaemia - ferritin low, continue iron",
    ])
    _add({
        "resourceType": "DocumentReference",
        "id": "doc-002",
        "status": "current",
        "type": {
            "coding": [{"system": "http://loinc.org", "code": "11506-3", "display": "Progress Note"}],
            "text": "Progress Note",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "date": "2025-04-15T10:00:00Z",
        "author": [{"display": "Dr. James Wilson"}],
        "description": "Follow-up visit progress note - April 2025",
        "content": [
            {
                "attachment": {
                    "contentType": "application/pdf",
                    "data": base64.b64encode(clinical_note_pdf).decode(),
                    "title": "Progress_Note_Martinez_20250415.pdf",
                }
            }
        ],
    })

    # 7c. Referral letter
    referral_pdf = _make_pdf("Referral Letter - Cardiology", [
        "Date: 2025-03-14",
        "From: Dr. James Wilson, Internal Medicine",
        "To: Dr. Sarah Chen, Cardiology",
        "",
        "RE: Elena Sofia Martinez, DOB 1978-03-15",
        "",
        "Dear Dr. Chen,",
        "",
        "I am referring this 47-year-old female patient for evaluation",
        "of new-onset exertional chest pain. She has a history of T2DM,",
        "HTN, and hyperlipidaemia. ECG shows non-specific ST changes.",
        "Troponin negative x2. Please evaluate for possible ACS.",
        "",
        "Thank you for your consultation.",
        "",
        "Dr. James Wilson",
    ])
    _add({
        "resourceType": "DocumentReference",
        "id": "doc-003",
        "status": "current",
        "type": {
            "coding": [{"system": "http://loinc.org", "code": "57133-1", "display": "Referral Note"}],
            "text": "Referral Note",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "date": "2025-03-14T14:00:00Z",
        "author": [{"display": "Dr. James Wilson"}],
        "description": "Referral to cardiology for chest pain evaluation",
        "content": [
            {
                "attachment": {
                    "contentType": "application/pdf",
                    "data": base64.b64encode(referral_pdf).decode(),
                    "title": "Referral_Cardiology_Martinez_20250314.pdf",
                }
            }
        ],
    })

    # 7d. Patient consent form scan (image attachment)
    consent_image = _make_png(r=240, g=240, b=230, w=200, h=280)  # off-white image simulating a scanned page
    _add({
        "resourceType": "DocumentReference",
        "id": "doc-004",
        "status": "current",
        "type": {
            "coding": [{"system": "http://loinc.org", "code": "59284-0", "display": "Consent Document"}],
            "text": "Consent Document",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "date": "2025-03-15T08:00:00Z",
        "author": [{"display": "Registration Desk"}],
        "description": "Scanned signed consent form for cardiac catheterisation",
        "content": [
            {
                "attachment": {
                    "contentType": "image/png",
                    "data": base64.b64encode(consent_image).decode(),
                    "title": "Consent_Cardiac_Cath_Martinez.png",
                }
            }
        ],
    })

    # ── 8. DiagnosticReports with presentedForm PDFs ───────────────────────
    # 8a. Lab report
    lab_report_pdf = _make_pdf("Laboratory Report - Elena Martinez", [
        "Report Date: 2025-04-10",
        "Ordering Physician: Dr. James Wilson",
        "Specimen: Blood (venipuncture)",
        "",
        "COMPLETE BLOOD COUNT:",
        "  Hemoglobin      11.2 g/dL     (L)  Ref: 12.0-16.0",
        "  Hematocrit      34.5 %        (L)  Ref: 36-46",
        "  WBC             7.2 10*3/uL        Ref: 4.5-11.0",
        "  Platelets       245 10*3/uL        Ref: 150-400",
        "",
        "BASIC METABOLIC PANEL:",
        "  Glucose(fasting) 142 mg/dL    (H)  Ref: 70-100",
        "  Creatinine      0.9 mg/dL          Ref: 0.6-1.2",
        "  BUN             18 mg/dL           Ref: 7-20",
        "  Sodium          140 mEq/L          Ref: 136-145",
        "  Potassium       4.3 mEq/L          Ref: 3.5-5.0",
        "",
        "HEMOGLOBIN A1C:   7.8 %         (H)  Ref: < 5.7",
        "",
        "LIPID PANEL:",
        "  Total Chol      215 mg/dL     (H)  Ref: < 200",
        "  Triglycerides   185 mg/dL     (H)  Ref: < 150",
        "  HDL             42 mg/dL      (L)  Ref: > 40",
        "  LDL (calc)      136 mg/dL     (H)  Ref: < 100",
        "",
        "IRON STUDIES:",
        "  Serum Iron      45 mcg/dL     (L)  Ref: 60-170",
        "  Ferritin        12 ng/mL      (L)  Ref: 12-150",
        "  TIBC            420 mcg/dL    (H)  Ref: 250-370",
        "",
        "Electronically signed: Lab Director",
    ])
    _add({
        "resourceType": "DiagnosticReport",
        "id": "diag-001",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB", "display": "Laboratory"}]}],
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "57021-8", "display": "CBC W Auto Differential panel + Metabolic Panel"}],
            "text": "Comprehensive Lab Panel",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "effectiveDateTime": "2025-04-10T08:00:00Z",
        "issued": "2025-04-10T14:30:00Z",
        "performer": [{"display": "Springfield General Hospital Lab"}],
        "conclusion": "Anaemia (iron deficiency pattern), poorly controlled diabetes (HbA1c 7.8%), hyperlipidaemia, renal function preserved.",
        "presentedForm": [
            {
                "contentType": "application/pdf",
                "data": base64.b64encode(lab_report_pdf).decode(),
                "title": "Lab_Report_Martinez_20250410.pdf",
            }
        ],
    })

    # 8b. Radiology report — Chest X-ray
    cxr_report_pdf = _make_pdf("Radiology Report - Chest X-ray", [
        "Patient: Elena Sofia Martinez  DOB: 1978-03-15",
        "Exam: PA and Lateral Chest X-ray",
        "Date: 2025-03-15",
        "Clinical History: Chest pain, rule out CHF/pneumonia",
        "",
        "FINDINGS:",
        "Heart size is at the upper limits of normal.",
        "Mild cardiomegaly cannot be excluded.",
        "Lungs are clear bilaterally without consolidation,",
        "effusion, or pneumothorax.",
        "Mediastinal contours are within normal limits.",
        "No acute bony abnormality.",
        "",
        "IMPRESSION:",
        "1. Borderline cardiac silhouette. Recommend echo for",
        "   further evaluation if not recently performed.",
        "2. No acute cardiopulmonary disease.",
        "",
        "Electronically signed: Dr. Lisa Park, Radiologist",
    ])
    cxr_image = _make_png(r=40, g=40, b=45, w=300, h=350)  # dark image simulating X-ray
    _add({
        "resourceType": "DiagnosticReport",
        "id": "diag-002",
        "status": "final",
        "category": [{"coding": [{"system": "http://loinc.org", "code": "LP29684-5", "display": "Radiology"}]}],
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "36643-5", "display": "Chest X-ray PA and Lateral"}],
            "text": "Chest X-ray PA and Lateral",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "effectiveDateTime": "2025-03-15T11:00:00Z",
        "issued": "2025-03-15T13:45:00Z",
        "performer": [{"display": "Dr. Lisa Park"}],
        "conclusion": "Borderline cardiac silhouette. No acute cardiopulmonary disease.",
        "presentedForm": [
            {
                "contentType": "application/pdf",
                "data": base64.b64encode(cxr_report_pdf).decode(),
                "title": "CXR_Report_Martinez_20250315.pdf",
            },
            {
                "contentType": "image/png",
                "data": base64.b64encode(cxr_image).decode(),
                "title": "CXR_Image_Martinez_20250315.png",
            },
        ],
    })

    # 8c. Cardiac catheterisation report
    cath_pdf = _make_pdf("Cardiac Catheterisation Report", [
        "Patient: Elena Sofia Martinez  DOB: 1978-03-15",
        "Date: 2025-03-16",
        "Operator: Dr. Sarah Chen, Interventional Cardiology",
        "",
        "PROCEDURE: Diagnostic left heart catheterisation",
        "and coronary angiography",
        "",
        "ACCESS: Right radial artery",
        "",
        "FINDINGS:",
        "Left Main: Patent, no stenosis",
        "LAD: 70% stenosis in mid-segment",
        "LCx: 30% stenosis in proximal segment",
        "RCA: Patent, no significant stenosis",
        "",
        "LV Function: EF estimated 55%",
        "LVEDP: 14 mmHg",
        "",
        "IMPRESSION: Single-vessel CAD with 70% mid-LAD stenosis.",
        "LV function preserved. Medical management recommended",
        "with close follow-up. PCI to be considered if symptoms",
        "recur despite optimal medical therapy.",
        "",
        "Dr. Sarah Chen",
    ])
    _add({
        "resourceType": "DiagnosticReport",
        "id": "diag-003",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "CUS", "display": "Cardiac Ultrasound"}]}],
        "code": {
            "coding": [{"system": "http://snomed.info/sct", "code": "41976001", "display": "Cardiac catheterisation"}],
            "text": "Cardiac Catheterisation Report",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "effectiveDateTime": "2025-03-16T10:00:00Z",
        "issued": "2025-03-16T15:00:00Z",
        "performer": [{"display": "Dr. Sarah Chen"}],
        "conclusion": "Single-vessel CAD with 70% mid-LAD stenosis. LV function preserved (EF 55%). Medical management recommended.",
        "presentedForm": [
            {
                "contentType": "application/pdf",
                "data": base64.b64encode(cath_pdf).decode(),
                "title": "Cardiac_Cath_Report_Martinez_20250316.pdf",
            }
        ],
    })

    # 8d. ECG report
    ecg_pdf = _make_pdf("ECG Report", [
        "Patient: Elena Sofia Martinez  DOB: 1978-03-15",
        "Date: 2025-03-15 09:15",
        "Indication: Chest pain",
        "",
        "FINDINGS:",
        "Rate: 78 bpm",
        "Rhythm: Normal sinus rhythm",
        "Axis: Normal",
        "PR Interval: 160 ms",
        "QRS Duration: 88 ms",
        "QTc: 420 ms",
        "",
        "ST-T Changes: Non-specific ST-T wave changes in leads",
        "V4-V6. No ST elevation or depression > 1mm.",
        "",
        "INTERPRETATION:",
        "Normal sinus rhythm with non-specific ST-T wave changes.",
        "Clinical correlation recommended.",
        "",
        "Dr. James Wilson",
    ])
    _add({
        "resourceType": "DiagnosticReport",
        "id": "diag-004",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "EC", "display": "Electrocardiac"}]}],
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "11524-6", "display": "ECG Study"}],
            "text": "12-Lead ECG",
        },
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "effectiveDateTime": "2025-03-15T09:15:00Z",
        "issued": "2025-03-15T09:30:00Z",
        "performer": [{"display": "Dr. James Wilson"}],
        "conclusion": "Normal sinus rhythm with non-specific ST-T wave changes in V4-V6.",
        "presentedForm": [
            {
                "contentType": "application/pdf",
                "data": base64.b64encode(ecg_pdf).decode(),
                "title": "ECG_Report_Martinez_20250315.pdf",
            }
        ],
    })

    # ── 9. ImagingStudy ────────────────────────────────────────────────────
    _add({
        "resourceType": "ImagingStudy",
        "id": "img-001",
        "status": "available",
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "started": "2025-03-15T11:00:00Z",
        "numberOfSeries": 2,
        "numberOfInstances": 4,
        "description": "Chest X-ray PA and Lateral",
        "modality": [{"system": "http://dicom.nema.org/resources/ontology/DCM", "code": "CR", "display": "Computed Radiography"}],
        "reasonCode": [{"text": "Chest pain, rule out CHF/pneumonia"}],
        "series": [
            {
                "uid": "1.2.3.4.5.6.7.8.1",
                "modality": {"system": "http://dicom.nema.org/resources/ontology/DCM", "code": "CR", "display": "Computed Radiography"},
                "bodySite": {"system": "http://snomed.info/sct", "code": "51185008", "display": "Thorax"},
                "numberOfInstances": 2,
                "description": "PA and Lateral views",
            },
        ],
    })

    # ── 10. Procedures ─────────────────────────────────────────────────────
    procedures = [
        ("proc-001", "41976001", "Cardiac catheterisation", "2025-03-16", "Dr. Sarah Chen"),
        ("proc-002", "29303009", "Electrocardiogram", "2025-03-15", "Dr. James Wilson"),
        ("proc-003", "399208008", "Chest X-ray", "2025-03-15", "Dr. Lisa Park"),
        ("proc-004", "271442007", "Phlebotomy (blood draw)", "2025-04-10", "Lab Technician"),
    ]
    for pid, snomed, display, performed, performer in procedures:
        _add({
            "resourceType": "Procedure",
            "id": pid,
            "status": "completed",
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "code": snomed, "display": display}],
                "text": display,
            },
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "performedDateTime": performed,
            "performer": [{"actor": {"display": performer}}],
        })

    # ── 11. CareTeam ───────────────────────────────────────────────────────
    _add({
        "resourceType": "CareTeam",
        "id": "careteam-001",
        "status": "active",
        "name": "Primary Care Team - Martinez",
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "participant": [
            {
                "role": [{"coding": [{"display": "Primary Care Physician"}]}],
                "member": {"display": "Dr. James Wilson"},
            },
            {
                "role": [{"coding": [{"display": "Cardiologist"}]}],
                "member": {"display": "Dr. Sarah Chen"},
            },
            {
                "role": [{"coding": [{"display": "Radiologist"}]}],
                "member": {"display": "Dr. Lisa Park"},
            },
            {
                "role": [{"coding": [{"display": "Clinical Pharmacist"}]}],
                "member": {"display": "Dr. Amy Rogers, PharmD"},
            },
            {
                "role": [{"coding": [{"display": "Nurse Practitioner"}]}],
                "member": {"display": "Maria Johnson, NP"},
            },
            {
                "role": [{"coding": [{"display": "Endocrinologist"}]}],
                "member": {"display": "Dr. Robert Kim"},
            },
        ],
    })

    # ── 12. Encounters ─────────────────────────────────────────────────────
    encounters = [
        ("enc-001", "IMP", "inpatient encounter", "2025-03-15", "2025-03-20", "Admission for chest pain evaluation and cardiac catheterisation"),
        ("enc-002", "AMB", "ambulatory", "2025-04-15", None, "Follow-up visit - post-cardiac catheterisation"),
        ("enc-003", "AMB", "ambulatory", "2025-01-10", None, "Routine diabetes and hypertension follow-up"),
        ("enc-004", "AMB", "ambulatory", "2024-10-15", None, "Annual physical examination"),
    ]
    for eid, cls_code, cls_display, start, end, reason in encounters:
        enc = {
            "resourceType": "Encounter",
            "id": eid,
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": cls_code, "display": cls_display},
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "period": {"start": start},
            "reasonCode": [{"text": reason}],
        }
        if end:
            enc["period"]["end"] = end
        _add(enc)

    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": entries,
    }


# ── POST the bundle to the FHIR server ────────────────────────────────────────

def load_data(fhir_url: str, token: str | None = None) -> None:
    bundle = build_bundle()
    total = len(bundle["entry"])
    print(f"\n  FHIR bundle built — {total} resources")
    print(f"  Target: {fhir_url}\n")

    # Optionally save the bundle to disk
    out_path = Path(__file__).parent / "patient_bundle.json"
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"  Bundle saved to {out_path}\n")

    headers = {"Content-Type": "application/fhir+json", "Accept": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = httpx.post(fhir_url, json=bundle, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        successes = sum(
            1 for e in result.get("entry", [])
            if str(e.get("response", {}).get("status", "")).startswith(("200", "201"))
        )
        print(f"  SUCCESS: {successes}/{total} resources created/updated")
        print(f"  Patient ID: {PATIENT_ID}")
        print(f"  URL: {fhir_url}/Patient/{PATIENT_ID}\n")
    except httpx.HTTPStatusError as e:
        print(f"  ERROR {e.response.status_code}: {e.response.text[:500]}")
        sys.exit(1)
    except Exception as e:
        print(f"  CONNECTION ERROR: {e}")
        print("  Is the FHIR server running? Try:")
        print("    docker run -p 8080:8080 hapiproject/hapi:latest")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Load test FHIR patient data")
    parser.add_argument("--fhir-url", default=DEFAULT_FHIR_URL, help="FHIR server base URL")
    parser.add_argument("--token", default=None, help="Bearer token for authentication")
    parser.add_argument("--save-only", action="store_true", help="Only save the bundle JSON, don't POST")
    args = parser.parse_args()

    if args.save_only:
        bundle = build_bundle()
        out_path = Path(__file__).parent / "patient_bundle.json"
        out_path.write_text(json.dumps(bundle, indent=2))
        print(f"\n  Bundle saved to {out_path} ({len(bundle['entry'])} resources)")
        print("  Use --fhir-url to POST to a FHIR server.\n")
        return

    load_data(args.fhir_url, args.token)


if __name__ == "__main__":
    main()
