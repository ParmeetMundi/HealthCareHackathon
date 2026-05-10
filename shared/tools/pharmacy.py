"""
Pharmacy tools — drug interaction checks and medication information.

These tools are used by the Pharmacist Agent for medication safety review.
They use built-in reference tables — no external API required.
"""
import json
import logging

from crewai.tools import tool as crewai_tool

logger = logging.getLogger(__name__)

# ── Drug interaction table ─────────────────────────────────────────────────────
# Each entry: (drug_a, drug_b) -> {severity, mechanism, recommendation}
# Normalised to lowercase. Lookup checks both directions.

_DRUG_INTERACTIONS: dict[tuple[str, str], dict] = {
    ("warfarin", "aspirin"): {
        "severity": "major",
        "mechanism": "Increased anticoagulant effect and bleeding risk due to additive antiplatelet and anticoagulant activity.",
        "recommendation": "Avoid combination if possible. If necessary, monitor INR closely and watch for signs of bleeding.",
    },
    ("warfarin", "ibuprofen"): {
        "severity": "major",
        "mechanism": "NSAIDs inhibit platelet function and may increase anticoagulant effect of warfarin. Risk of GI bleeding.",
        "recommendation": "Avoid concurrent use. Consider acetaminophen as an alternative analgesic.",
    },
    ("warfarin", "amiodarone"): {
        "severity": "major",
        "mechanism": "Amiodarone inhibits CYP2C9, increasing warfarin levels and bleeding risk.",
        "recommendation": "Reduce warfarin dose by 30-50% when initiating amiodarone. Monitor INR frequently.",
    },
    ("warfarin", "fluconazole"): {
        "severity": "major",
        "mechanism": "Fluconazole inhibits CYP2C9 and CYP3A4, significantly increasing warfarin effect.",
        "recommendation": "Reduce warfarin dose and monitor INR closely during and after fluconazole therapy.",
    },
    ("warfarin", "metronidazole"): {
        "severity": "major",
        "mechanism": "Metronidazole inhibits CYP2C9, increasing warfarin anticoagulant effect.",
        "recommendation": "Monitor INR closely. Consider dose reduction of warfarin.",
    },
    ("warfarin", "omeprazole"): {
        "severity": "moderate",
        "mechanism": "Omeprazole may inhibit CYP2C19, potentially affecting warfarin metabolism.",
        "recommendation": "Monitor INR when starting or stopping omeprazole.",
    },
    ("metformin", "contrast dye"): {
        "severity": "major",
        "mechanism": "Iodinated contrast media may cause acute kidney injury, increasing risk of metformin-associated lactic acidosis.",
        "recommendation": "Withhold metformin 48 hours before and after contrast administration. Check renal function.",
    },
    ("metformin", "alcohol"): {
        "severity": "major",
        "mechanism": "Alcohol potentiates the effect of metformin on lactate metabolism, increasing lactic acidosis risk.",
        "recommendation": "Advise patient to limit alcohol intake. Monitor for symptoms of lactic acidosis.",
    },
    ("lisinopril", "potassium"): {
        "severity": "major",
        "mechanism": "ACE inhibitors reduce aldosterone, causing potassium retention. Supplemental potassium increases hyperkalemia risk.",
        "recommendation": "Monitor serum potassium regularly. Avoid potassium supplements unless clearly indicated.",
    },
    ("lisinopril", "spironolactone"): {
        "severity": "major",
        "mechanism": "Both drugs cause potassium retention, significantly increasing hyperkalemia risk.",
        "recommendation": "If combination is necessary, monitor potassium closely and start at low doses.",
    },
    ("lisinopril", "ibuprofen"): {
        "severity": "moderate",
        "mechanism": "NSAIDs may reduce the antihypertensive effect of ACE inhibitors and increase renal impairment risk.",
        "recommendation": "Monitor blood pressure and renal function. Consider alternative analgesic.",
    },
    ("simvastatin", "amiodarone"): {
        "severity": "major",
        "mechanism": "Amiodarone inhibits CYP3A4, increasing simvastatin levels and risk of rhabdomyolysis.",
        "recommendation": "Do not exceed simvastatin 20mg daily with amiodarone. Consider alternative statin.",
    },
    ("simvastatin", "amlodipine"): {
        "severity": "moderate",
        "mechanism": "Amlodipine inhibits CYP3A4, increasing simvastatin exposure and myopathy risk.",
        "recommendation": "Do not exceed simvastatin 20mg daily with amlodipine.",
    },
    ("simvastatin", "clarithromycin"): {
        "severity": "major",
        "mechanism": "Clarithromycin strongly inhibits CYP3A4, dramatically increasing simvastatin levels.",
        "recommendation": "Contraindicated. Suspend simvastatin during clarithromycin therapy or use azithromycin.",
    },
    ("atorvastatin", "clarithromycin"): {
        "severity": "major",
        "mechanism": "Clarithromycin inhibits CYP3A4, increasing atorvastatin levels and myopathy risk.",
        "recommendation": "Use lowest effective atorvastatin dose or consider azithromycin as alternative.",
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "major",
        "mechanism": "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel to its active metabolite.",
        "recommendation": "Use pantoprazole instead of omeprazole. Avoid esomeprazole as well.",
    },
    ("clopidogrel", "aspirin"): {
        "severity": "moderate",
        "mechanism": "Dual antiplatelet therapy increases bleeding risk.",
        "recommendation": "Combination is often clinically indicated (e.g. post-PCI). Monitor for bleeding. Use PPI gastroprotection.",
    },
    ("digoxin", "amiodarone"): {
        "severity": "major",
        "mechanism": "Amiodarone increases digoxin levels by reducing renal and non-renal clearance.",
        "recommendation": "Reduce digoxin dose by 50% when starting amiodarone. Monitor digoxin levels.",
    },
    ("digoxin", "furosemide"): {
        "severity": "major",
        "mechanism": "Furosemide-induced hypokalemia increases sensitivity to digoxin toxicity.",
        "recommendation": "Monitor potassium levels. Supplement potassium as needed. Watch for digoxin toxicity signs.",
    },
    ("digoxin", "verapamil"): {
        "severity": "major",
        "mechanism": "Verapamil increases digoxin serum levels by reducing renal clearance and additive AV node suppression.",
        "recommendation": "Reduce digoxin dose by 25-50%. Monitor digoxin levels and heart rate.",
    },
    ("metoprolol", "verapamil"): {
        "severity": "major",
        "mechanism": "Both drugs have negative chronotropic and inotropic effects, risk of severe bradycardia and heart block.",
        "recommendation": "Avoid combination. If necessary, use with extreme caution and continuous monitoring.",
    },
    ("metoprolol", "fluoxetine"): {
        "severity": "moderate",
        "mechanism": "Fluoxetine inhibits CYP2D6, increasing metoprolol levels and risk of bradycardia/hypotension.",
        "recommendation": "Monitor heart rate and blood pressure. Consider dose adjustment of metoprolol.",
    },
    ("ciprofloxacin", "theophylline"): {
        "severity": "major",
        "mechanism": "Ciprofloxacin inhibits CYP1A2, increasing theophylline levels and toxicity risk.",
        "recommendation": "Monitor theophylline levels. Consider reducing theophylline dose by 30-50%.",
    },
    ("ciprofloxacin", "antacids"): {
        "severity": "moderate",
        "mechanism": "Antacids containing aluminium/magnesium chelate ciprofloxacin, reducing absorption.",
        "recommendation": "Administer ciprofloxacin at least 2 hours before or 6 hours after antacids.",
    },
    ("ssri", "tramadol"): {
        "severity": "major",
        "mechanism": "Both drugs increase serotonin levels, risk of serotonin syndrome.",
        "recommendation": "Avoid combination if possible. Monitor for signs of serotonin syndrome (agitation, hyperthermia, clonus).",
    },
    ("ssri", "maoi"): {
        "severity": "major",
        "mechanism": "Extremely high serotonin levels leading to potentially fatal serotonin syndrome.",
        "recommendation": "Contraindicated. Allow adequate washout period (at least 14 days for MAOIs, 5 weeks for fluoxetine).",
    },
    ("fluoxetine", "tramadol"): {
        "severity": "major",
        "mechanism": "Serotonergic activity of both drugs increases risk of serotonin syndrome. Fluoxetine also inhibits CYP2D6.",
        "recommendation": "Avoid combination. Use alternative analgesic or antidepressant.",
    },
    ("sertraline", "tramadol"): {
        "severity": "major",
        "mechanism": "Both drugs increase serotonin levels, risk of serotonin syndrome.",
        "recommendation": "Avoid combination if possible. Monitor for serotonin syndrome symptoms.",
    },
    ("insulin", "beta-blockers"): {
        "severity": "moderate",
        "mechanism": "Beta-blockers may mask symptoms of hypoglycemia (tachycardia, tremor) and impair glycogenolysis.",
        "recommendation": "Use cardioselective beta-blockers. Educate patient about altered hypoglycemia symptoms.",
    },
    ("lithium", "ibuprofen"): {
        "severity": "major",
        "mechanism": "NSAIDs reduce renal lithium clearance, increasing lithium levels and toxicity risk.",
        "recommendation": "Monitor lithium levels closely. Consider acetaminophen as alternative analgesic.",
    },
    ("lithium", "lisinopril"): {
        "severity": "major",
        "mechanism": "ACE inhibitors reduce renal lithium clearance, increasing lithium levels.",
        "recommendation": "Monitor lithium levels closely when starting or adjusting ACE inhibitor dose.",
    },
    ("levothyroxine", "calcium"): {
        "severity": "moderate",
        "mechanism": "Calcium supplements reduce levothyroxine absorption by forming insoluble complexes.",
        "recommendation": "Separate administration by at least 4 hours.",
    },
    ("levothyroxine", "iron"): {
        "severity": "moderate",
        "mechanism": "Iron supplements reduce levothyroxine absorption.",
        "recommendation": "Separate administration by at least 4 hours. Monitor TSH levels.",
    },
}

# ── Medication info table ──────────────────────────────────────────────────────

_MEDICATION_INFO: dict[str, dict] = {
    "metformin": {
        "drug_class": "Biguanide (antidiabetic)",
        "standard_dose_range": "500mg-2000mg daily in divided doses",
        "common_side_effects": ["nausea", "diarrhoea", "abdominal pain", "metallic taste"],
        "serious_side_effects": ["lactic acidosis (rare)", "vitamin B12 deficiency"],
        "contraindications": ["eGFR < 30 mL/min", "acute or chronic metabolic acidosis", "severe hepatic impairment"],
    },
    "lisinopril": {
        "drug_class": "ACE inhibitor (antihypertensive)",
        "standard_dose_range": "2.5mg-40mg once daily",
        "common_side_effects": ["dry cough", "dizziness", "headache", "hyperkalaemia"],
        "serious_side_effects": ["angioedema", "renal impairment", "hypotension"],
        "contraindications": ["history of angioedema", "bilateral renal artery stenosis", "pregnancy"],
    },
    "atorvastatin": {
        "drug_class": "HMG-CoA reductase inhibitor (statin)",
        "standard_dose_range": "10mg-80mg once daily",
        "common_side_effects": ["myalgia", "headache", "GI disturbance", "elevated liver enzymes"],
        "serious_side_effects": ["rhabdomyolysis", "hepatotoxicity", "new-onset diabetes"],
        "contraindications": ["active liver disease", "unexplained persistent transaminase elevation", "pregnancy"],
    },
    "amlodipine": {
        "drug_class": "Calcium channel blocker (antihypertensive)",
        "standard_dose_range": "2.5mg-10mg once daily",
        "common_side_effects": ["peripheral oedema", "flushing", "headache", "dizziness"],
        "serious_side_effects": ["severe hypotension", "worsening angina (rare)"],
        "contraindications": ["severe aortic stenosis", "unstable angina", "cardiogenic shock"],
    },
    "omeprazole": {
        "drug_class": "Proton pump inhibitor",
        "standard_dose_range": "10mg-40mg once daily",
        "common_side_effects": ["headache", "nausea", "abdominal pain", "diarrhoea"],
        "serious_side_effects": ["C. difficile infection", "hypomagnesaemia", "bone fractures (long-term)"],
        "contraindications": ["hypersensitivity to PPIs"],
    },
    "warfarin": {
        "drug_class": "Vitamin K antagonist (anticoagulant)",
        "standard_dose_range": "1mg-10mg daily (titrated to INR target 2.0-3.0)",
        "common_side_effects": ["bleeding", "bruising"],
        "serious_side_effects": ["major haemorrhage", "skin necrosis", "purple toe syndrome"],
        "contraindications": ["active bleeding", "haemorrhagic stroke", "pregnancy", "severe hepatic disease"],
    },
    "metoprolol": {
        "drug_class": "Beta-1 selective adrenergic blocker",
        "standard_dose_range": "25mg-200mg twice daily (tartrate) or 25mg-400mg once daily (succinate)",
        "common_side_effects": ["fatigue", "bradycardia", "dizziness", "cold extremities"],
        "serious_side_effects": ["severe bradycardia", "heart block", "bronchospasm", "heart failure exacerbation"],
        "contraindications": ["sinus bradycardia", "heart block > first degree", "cardiogenic shock", "decompensated HF"],
    },
    "furosemide": {
        "drug_class": "Loop diuretic",
        "standard_dose_range": "20mg-80mg daily (up to 600mg in severe oedema)",
        "common_side_effects": ["hypokalemia", "dehydration", "dizziness", "hyperuricaemia"],
        "serious_side_effects": ["severe electrolyte depletion", "ototoxicity", "pancreatitis"],
        "contraindications": ["anuria", "severe hyponatraemia", "severe hypokalaemia"],
    },
    "aspirin": {
        "drug_class": "Non-steroidal anti-inflammatory / antiplatelet",
        "standard_dose_range": "75mg-325mg daily (antiplatelet); 300mg-1000mg (analgesic)",
        "common_side_effects": ["GI irritation", "dyspepsia", "increased bleeding time"],
        "serious_side_effects": ["GI bleeding", "haemorrhagic stroke", "Reye syndrome (children)"],
        "contraindications": ["active GI bleeding", "aspirin-sensitive asthma", "children under 16 (Reye syndrome)"],
    },
    "clopidogrel": {
        "drug_class": "P2Y12 platelet inhibitor (antiplatelet)",
        "standard_dose_range": "75mg once daily (300-600mg loading dose)",
        "common_side_effects": ["bleeding", "bruising", "dyspepsia", "diarrhoea"],
        "serious_side_effects": ["major bleeding", "thrombotic thrombocytopenic purpura (TTP)"],
        "contraindications": ["active pathological bleeding", "severe hepatic impairment"],
    },
    "insulin": {
        "drug_class": "Hormone (antidiabetic)",
        "standard_dose_range": "Varies by type and patient; typically 0.5-1.0 units/kg/day total",
        "common_side_effects": ["hypoglycaemia", "weight gain", "injection site reactions"],
        "serious_side_effects": ["severe hypoglycaemia", "hypokalaemia", "lipodystrophy"],
        "contraindications": ["hypoglycaemia"],
    },
    "gabapentin": {
        "drug_class": "Gabapentinoid (anticonvulsant / neuropathic pain)",
        "standard_dose_range": "300mg-3600mg daily in three divided doses",
        "common_side_effects": ["somnolence", "dizziness", "ataxia", "peripheral oedema"],
        "serious_side_effects": ["respiratory depression (with opioids)", "suicidal ideation", "angioedema"],
        "contraindications": ["hypersensitivity to gabapentin"],
    },
    "sertraline": {
        "drug_class": "Selective serotonin reuptake inhibitor (SSRI)",
        "standard_dose_range": "50mg-200mg once daily",
        "common_side_effects": ["nausea", "diarrhoea", "insomnia", "sexual dysfunction", "headache"],
        "serious_side_effects": ["serotonin syndrome", "suicidal ideation (young adults)", "hyponatraemia"],
        "contraindications": ["concurrent MAOI use", "pimozide co-administration"],
    },
    "prednisone": {
        "drug_class": "Corticosteroid",
        "standard_dose_range": "5mg-60mg daily depending on indication",
        "common_side_effects": ["increased appetite", "insomnia", "mood changes", "fluid retention"],
        "serious_side_effects": ["adrenal suppression", "osteoporosis", "hyperglycaemia", "immunosuppression"],
        "contraindications": ["systemic fungal infections", "live vaccines during high-dose therapy"],
    },
    "amoxicillin": {
        "drug_class": "Aminopenicillin (antibiotic)",
        "standard_dose_range": "250mg-500mg three times daily or 875mg twice daily",
        "common_side_effects": ["diarrhoea", "nausea", "rash"],
        "serious_side_effects": ["anaphylaxis", "C. difficile colitis", "hepatic cholestasis"],
        "contraindications": ["penicillin allergy", "history of amoxicillin-associated hepatic dysfunction"],
    },
    "ciprofloxacin": {
        "drug_class": "Fluoroquinolone (antibiotic)",
        "standard_dose_range": "250mg-750mg twice daily",
        "common_side_effects": ["nausea", "diarrhoea", "headache", "dizziness"],
        "serious_side_effects": ["tendon rupture", "QT prolongation", "peripheral neuropathy", "aortic dissection"],
        "contraindications": ["concurrent tizanidine", "history of tendon disorder with fluoroquinolone"],
    },
    "pantoprazole": {
        "drug_class": "Proton pump inhibitor",
        "standard_dose_range": "20mg-40mg once daily",
        "common_side_effects": ["headache", "diarrhoea", "nausea", "abdominal pain"],
        "serious_side_effects": ["C. difficile infection", "hypomagnesaemia", "bone fractures (long-term)"],
        "contraindications": ["hypersensitivity to PPIs", "co-administration with rilpivirine"],
    },
    "levothyroxine": {
        "drug_class": "Thyroid hormone replacement",
        "standard_dose_range": "25mcg-200mcg once daily (titrated to TSH)",
        "common_side_effects": ["palpitations", "tremor", "insomnia", "weight loss (if over-replaced)"],
        "serious_side_effects": ["angina", "arrhythmia", "osteoporosis (chronic excess)"],
        "contraindications": ["untreated adrenal insufficiency", "acute myocardial infarction"],
    },
    "hydrochlorothiazide": {
        "drug_class": "Thiazide diuretic",
        "standard_dose_range": "12.5mg-50mg once daily",
        "common_side_effects": ["hypokalaemia", "hyperuricaemia", "dizziness", "photosensitivity"],
        "serious_side_effects": ["severe electrolyte imbalance", "pancreatitis", "skin cancer (long-term)"],
        "contraindications": ["anuria", "severe renal impairment", "Addison's disease"],
    },
    "losartan": {
        "drug_class": "Angiotensin II receptor blocker (ARB)",
        "standard_dose_range": "25mg-100mg once daily",
        "common_side_effects": ["dizziness", "hyperkalaemia", "fatigue"],
        "serious_side_effects": ["angioedema (rare)", "renal impairment", "hypotension"],
        "contraindications": ["pregnancy", "bilateral renal artery stenosis"],
    },
    "simvastatin": {
        "drug_class": "HMG-CoA reductase inhibitor (statin)",
        "standard_dose_range": "10mg-40mg once daily at bedtime (max 80mg only if tolerated >12 months)",
        "common_side_effects": ["myalgia", "headache", "GI disturbance", "elevated liver enzymes"],
        "serious_side_effects": ["rhabdomyolysis", "hepatotoxicity"],
        "contraindications": ["active liver disease", "pregnancy", "concurrent strong CYP3A4 inhibitors"],
    },
}


def _normalize_drug_name(name: str) -> str:
    """Normalize drug name for lookup."""
    return name.strip().lower().replace("-", " ").replace("_", " ")


@crewai_tool("Check Drug Interactions")
def check_drug_interactions(drug_a: str, drug_b: str) -> str:
    """
    Check for known drug-drug interactions between two medications.
    Uses a built-in reference table of 30+ common interaction pairs.

    Args:
        drug_a: First medication name.
        drug_b: Second medication name.
    """
    a = _normalize_drug_name(drug_a)
    b = _normalize_drug_name(drug_b)

    logger.info("tool_check_drug_interactions drug_a=%s drug_b=%s", a, b)

    # Check both orderings
    interaction = _DRUG_INTERACTIONS.get((a, b)) or _DRUG_INTERACTIONS.get((b, a))

    if interaction:
        return json.dumps({
            "status": "interaction_found",
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": interaction["severity"],
            "mechanism": interaction["mechanism"],
            "recommendation": interaction["recommendation"],
        })

    # Partial match — check if either drug name is a substring
    for (da, db), info in _DRUG_INTERACTIONS.items():
        if (a in da or da in a) and (b in db or db in b):
            return json.dumps({
                "status": "interaction_found",
                "drug_a": drug_a,
                "drug_b": drug_b,
                "matched_pair": f"{da} + {db}",
                "severity": info["severity"],
                "mechanism": info["mechanism"],
                "recommendation": info["recommendation"],
                "note": "Matched via partial name matching.",
            })
        if (a in db or db in a) and (b in da or da in b):
            return json.dumps({
                "status": "interaction_found",
                "drug_a": drug_a,
                "drug_b": drug_b,
                "matched_pair": f"{da} + {db}",
                "severity": info["severity"],
                "mechanism": info["mechanism"],
                "recommendation": info["recommendation"],
                "note": "Matched via partial name matching.",
            })

    return json.dumps({
        "status": "no_interaction_found",
        "drug_a": drug_a,
        "drug_b": drug_b,
        "note": "No known interaction found in the built-in reference table. This does not guarantee safety — consult a pharmacist or comprehensive drug interaction database.",
    })


@crewai_tool("Get Medication Info")
def get_medication_info(drug_name: str) -> str:
    """
    Get standard medication information including drug class, dosage range,
    side effects, and contraindications from the built-in reference table.

    Args:
        drug_name: Name of the medication to look up.
    """
    name = _normalize_drug_name(drug_name)
    logger.info("tool_get_medication_info drug=%s", name)

    info = _MEDICATION_INFO.get(name)
    if info:
        return json.dumps({
            "status": "found",
            "drug_name": drug_name,
            **info,
        })

    # Partial match
    for key, info in _MEDICATION_INFO.items():
        if name in key or key in name:
            return json.dumps({
                "status": "found",
                "drug_name": drug_name,
                "matched_name": key,
                "note": "Matched via partial name matching.",
                **info,
            })

    return json.dumps({
        "status": "not_found",
        "drug_name": drug_name,
        "note": "Medication not found in built-in reference table.",
        "available_medications": sorted(_MEDICATION_INFO.keys()),
    })
