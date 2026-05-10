"""
Pharmacy tools — drug interaction checks and medication information.

These tools are used by the Pharmacist Agent for medication safety review.
They use built-in reference tables for verified common interactions and fall
back to LLM-powered pharmacological analysis for any drug pair or medication
not covered by the local tables.
"""
import json
import logging
import os

import litellm
from crewai.tools import tool as crewai_tool

logger = logging.getLogger(__name__)

_LLM_MODEL = os.getenv("CREWAI_MODEL", "gpt-4o-mini")

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
    # ── Opioid interactions ────────────────────────────────────────────────────
    ("opioids", "benzodiazepines"): {
        "severity": "major",
        "mechanism": "Combined CNS depression increases risk of respiratory depression, sedation, coma, and death.",
        "recommendation": "Avoid concurrent use. If necessary, use lowest effective doses and monitor respiratory status closely.",
    },
    ("morphine", "benzodiazepines"): {
        "severity": "major",
        "mechanism": "Additive CNS and respiratory depression. FDA black box warning.",
        "recommendation": "Avoid combination. If unavoidable, limit dosages and duration. Monitor for respiratory depression.",
    },
    ("oxycodone", "benzodiazepines"): {
        "severity": "major",
        "mechanism": "Additive CNS and respiratory depression. FDA black box warning.",
        "recommendation": "Avoid combination. If unavoidable, limit dosages and duration. Monitor for respiratory depression.",
    },
    ("tramadol", "gabapentin"): {
        "severity": "major",
        "mechanism": "Additive CNS depression and increased seizure risk.",
        "recommendation": "Use with caution. Monitor for excessive sedation and respiratory depression.",
    },
    ("morphine", "gabapentin"): {
        "severity": "major",
        "mechanism": "Gabapentin may increase morphine exposure and additive CNS depression.",
        "recommendation": "Reduce opioid dose. Monitor for respiratory depression and excessive sedation.",
    },
    # ── DOAC / anticoagulant interactions ──────────────────────────────────────
    ("apixaban", "ketoconazole"): {
        "severity": "major",
        "mechanism": "Ketoconazole strongly inhibits CYP3A4 and P-gp, significantly increasing apixaban levels.",
        "recommendation": "Avoid combination. If unavoidable, reduce apixaban dose by 50%.",
    },
    ("apixaban", "rifampin"): {
        "severity": "major",
        "mechanism": "Rifampin induces CYP3A4 and P-gp, significantly reducing apixaban levels and efficacy.",
        "recommendation": "Avoid concurrent use. Consider alternative anticoagulant or anti-TB regimen.",
    },
    ("rivaroxaban", "ketoconazole"): {
        "severity": "major",
        "mechanism": "Ketoconazole inhibits CYP3A4 and P-gp, markedly increasing rivaroxaban levels and bleeding risk.",
        "recommendation": "Avoid concurrent use.",
    },
    ("rivaroxaban", "aspirin"): {
        "severity": "major",
        "mechanism": "Additive bleeding risk from combined anticoagulant and antiplatelet effects.",
        "recommendation": "Avoid unless clinically indicated (e.g. ACS). Use lowest aspirin dose and monitor for bleeding.",
    },
    ("apixaban", "aspirin"): {
        "severity": "major",
        "mechanism": "Additive bleeding risk from combined anticoagulant and antiplatelet effects.",
        "recommendation": "Avoid unless clinically indicated. Use lowest aspirin dose and monitor for bleeding.",
    },
    ("dabigatran", "verapamil"): {
        "severity": "major",
        "mechanism": "Verapamil inhibits P-gp, increasing dabigatran levels and bleeding risk.",
        "recommendation": "Reduce dabigatran dose. Administer dabigatran at least 2 hours before verapamil.",
    },
    ("enoxaparin", "aspirin"): {
        "severity": "major",
        "mechanism": "Additive anticoagulant and antiplatelet effects increase bleeding risk.",
        "recommendation": "Use combination only when clinically indicated. Monitor for signs of bleeding.",
    },
    # ── Methotrexate interactions ──────────────────────────────────────────────
    ("methotrexate", "ibuprofen"): {
        "severity": "major",
        "mechanism": "NSAIDs reduce renal clearance of methotrexate, increasing toxicity risk (myelosuppression, nephrotoxicity).",
        "recommendation": "Avoid NSAIDs with high-dose methotrexate. Monitor closely with low-dose MTX.",
    },
    ("methotrexate", "trimethoprim"): {
        "severity": "major",
        "mechanism": "Both drugs are folate antagonists. Increased risk of pancytopenia and bone marrow suppression.",
        "recommendation": "Avoid combination. If unavoidable, monitor CBC closely and supplement with folic acid.",
    },
    ("methotrexate", "omeprazole"): {
        "severity": "moderate",
        "mechanism": "PPIs may delay renal elimination of methotrexate, increasing exposure and toxicity risk.",
        "recommendation": "Consider withholding PPI during high-dose methotrexate. Monitor methotrexate levels.",
    },
    # ── Antidepressant / psychiatric interactions ──────────────────────────────
    ("venlafaxine", "tramadol"): {
        "severity": "major",
        "mechanism": "Both drugs increase serotonin levels, risk of serotonin syndrome. Both are CYP2D6 substrates.",
        "recommendation": "Avoid combination. Use alternative analgesic.",
    },
    ("duloxetine", "tramadol"): {
        "severity": "major",
        "mechanism": "Serotonergic activity of both drugs increases risk of serotonin syndrome.",
        "recommendation": "Avoid combination if possible. Monitor for serotonin syndrome symptoms.",
    },
    ("escitalopram", "tramadol"): {
        "severity": "major",
        "mechanism": "Both drugs increase serotonin levels, risk of serotonin syndrome.",
        "recommendation": "Avoid combination. Use alternative analgesic.",
    },
    ("fluoxetine", "maoi"): {
        "severity": "major",
        "mechanism": "Extremely high serotonin levels leading to potentially fatal serotonin syndrome.",
        "recommendation": "Contraindicated. Allow 5-week washout after stopping fluoxetine before starting MAOI.",
    },
    ("lithium", "furosemide"): {
        "severity": "major",
        "mechanism": "Furosemide-induced sodium and volume depletion increases lithium reabsorption and toxicity risk.",
        "recommendation": "Monitor lithium levels closely. Maintain adequate hydration and sodium intake.",
    },
    ("lithium", "hydrochlorothiazide"): {
        "severity": "major",
        "mechanism": "Thiazides reduce renal lithium clearance by 25%, increasing lithium levels and toxicity risk.",
        "recommendation": "Reduce lithium dose by 25-50% when initiating thiazide. Monitor lithium levels closely.",
    },
    # ── Diabetes medication interactions ───────────────────────────────────────
    ("metformin", "furosemide"): {
        "severity": "moderate",
        "mechanism": "Furosemide may increase metformin levels. Both can affect renal function.",
        "recommendation": "Monitor renal function and blood glucose. Adjust metformin dose if needed.",
    },
    ("glipizide", "fluconazole"): {
        "severity": "major",
        "mechanism": "Fluconazole inhibits CYP2C9, increasing glipizide levels and hypoglycaemia risk.",
        "recommendation": "Monitor blood glucose closely. Consider reducing glipizide dose.",
    },
    ("insulin", "ace inhibitors"): {
        "severity": "moderate",
        "mechanism": "ACE inhibitors may enhance insulin sensitivity and increase hypoglycaemia risk.",
        "recommendation": "Monitor blood glucose when initiating ACE inhibitor. Adjust insulin dose if needed.",
    },
    ("empagliflozin", "furosemide"): {
        "severity": "moderate",
        "mechanism": "Both drugs cause volume depletion, increasing risk of dehydration and hypotension.",
        "recommendation": "Assess volume status before initiating. Monitor blood pressure and renal function.",
    },
    # ── Cardiovascular interactions ────────────────────────────────────────────
    ("diltiazem", "simvastatin"): {
        "severity": "major",
        "mechanism": "Diltiazem inhibits CYP3A4, increasing simvastatin levels and rhabdomyolysis risk.",
        "recommendation": "Do not exceed simvastatin 10mg daily with diltiazem. Consider alternative statin.",
    },
    ("diltiazem", "metoprolol"): {
        "severity": "major",
        "mechanism": "Additive negative chronotropic and dromotropic effects, risk of severe bradycardia and heart block.",
        "recommendation": "Avoid combination if possible. If necessary, monitor ECG and heart rate closely.",
    },
    ("amiodarone", "metformin"): {
        "severity": "moderate",
        "mechanism": "Amiodarone may cause thyroid dysfunction, complicating diabetes management.",
        "recommendation": "Monitor thyroid function and blood glucose regularly.",
    },
    ("amlodipine", "simvastatin"): {
        "severity": "moderate",
        "mechanism": "Amlodipine inhibits CYP3A4, increasing simvastatin exposure and myopathy risk.",
        "recommendation": "Do not exceed simvastatin 20mg daily with amlodipine.",
    },
    ("losartan", "potassium"): {
        "severity": "major",
        "mechanism": "ARBs reduce aldosterone causing potassium retention. Supplemental potassium increases hyperkalemia risk.",
        "recommendation": "Monitor serum potassium regularly. Avoid potassium supplements unless clearly indicated.",
    },
    ("losartan", "spironolactone"): {
        "severity": "major",
        "mechanism": "Both drugs cause potassium retention, significantly increasing hyperkalemia risk.",
        "recommendation": "If combination is necessary, monitor potassium closely and start at low doses.",
    },
    ("lisinopril", "losartan"): {
        "severity": "major",
        "mechanism": "Dual RAAS blockade increases risk of hypotension, hyperkalemia, and renal impairment.",
        "recommendation": "Avoid combination. No benefit shown in most patient populations.",
    },
    # ── Antibiotic interactions ────────────────────────────────────────────────
    ("azithromycin", "amiodarone"): {
        "severity": "major",
        "mechanism": "Both drugs prolong QT interval, increasing risk of torsades de pointes.",
        "recommendation": "Avoid combination. Use alternative antibiotic without QT prolongation risk.",
    },
    ("clarithromycin", "amiodarone"): {
        "severity": "major",
        "mechanism": "Both prolong QT interval. Clarithromycin also inhibits CYP3A4, affecting amiodarone metabolism.",
        "recommendation": "Avoid combination. Use azithromycin with caution or choose alternative antibiotic.",
    },
    ("ciprofloxacin", "tizanidine"): {
        "severity": "major",
        "mechanism": "Ciprofloxacin inhibits CYP1A2, increasing tizanidine levels up to 10-fold, causing severe hypotension and sedation.",
        "recommendation": "Contraindicated. Use alternative antibiotic or muscle relaxant.",
    },
    ("doxycycline", "antacids"): {
        "severity": "moderate",
        "mechanism": "Divalent cations in antacids chelate doxycycline, reducing absorption by up to 80%.",
        "recommendation": "Separate administration by at least 2-3 hours.",
    },
    ("doxycycline", "iron"): {
        "severity": "moderate",
        "mechanism": "Iron forms insoluble chelates with doxycycline, reducing absorption.",
        "recommendation": "Separate administration by at least 2-3 hours.",
    },
    ("amoxicillin", "methotrexate"): {
        "severity": "major",
        "mechanism": "Amoxicillin may reduce renal clearance of methotrexate, increasing toxicity risk.",
        "recommendation": "Monitor methotrexate levels and renal function closely.",
    },
    # ── Miscellaneous common interactions ──────────────────────────────────────
    ("sildenafil", "nitrates"): {
        "severity": "major",
        "mechanism": "Both cause vasodilation via NO/cGMP pathway, risk of severe life-threatening hypotension.",
        "recommendation": "Contraindicated. Do not use within 24 hours of nitrate administration (48 hours for tadalafil).",
    },
    ("potassium", "spironolactone"): {
        "severity": "major",
        "mechanism": "Spironolactone is a potassium-sparing diuretic. Supplemental potassium significantly increases hyperkalemia risk.",
        "recommendation": "Avoid potassium supplements unless clearly indicated. Monitor serum potassium regularly.",
    },
    ("albuterol", "propranolol"): {
        "severity": "major",
        "mechanism": "Non-selective beta-blockers antagonize bronchodilator effect of albuterol and may cause bronchospasm.",
        "recommendation": "Avoid non-selective beta-blockers in asthma/COPD. Use cardioselective beta-blocker if needed.",
    },
    ("tamsulosin", "sildenafil"): {
        "severity": "moderate",
        "mechanism": "Additive vasodilatory and alpha-blocking effects, risk of orthostatic hypotension.",
        "recommendation": "Start sildenafil at lowest dose. Advise patient about orthostatic precautions.",
    },
    ("ondansetron", "amiodarone"): {
        "severity": "major",
        "mechanism": "Both drugs prolong QT interval, increasing risk of torsades de pointes.",
        "recommendation": "Avoid combination. Use alternative antiemetic (e.g. metoclopramide).",
    },
    ("prednisone", "ibuprofen"): {
        "severity": "moderate",
        "mechanism": "Additive risk of GI bleeding and peptic ulceration.",
        "recommendation": "Use combination with caution. Consider PPI gastroprotection. Monitor for GI symptoms.",
    },
    ("prednisone", "insulin"): {
        "severity": "moderate",
        "mechanism": "Corticosteroids cause hyperglycaemia by increasing insulin resistance and hepatic gluconeogenesis.",
        "recommendation": "Monitor blood glucose frequently. Increase insulin dose as needed during steroid therapy.",
    },
    ("carbamazepine", "oral contraceptives"): {
        "severity": "major",
        "mechanism": "Carbamazepine induces CYP3A4, reducing contraceptive hormone levels and efficacy.",
        "recommendation": "Use alternative contraception (e.g. IUD, depot injection). Do not rely on oral contraceptives alone.",
    },
    ("phenytoin", "warfarin"): {
        "severity": "major",
        "mechanism": "Complex interaction: phenytoin initially displaces warfarin from protein binding, then induces metabolism.",
        "recommendation": "Monitor INR closely when starting, adjusting, or stopping phenytoin. Frequent dose adjustments needed.",
    },
    ("valproic acid", "lamotrigine"): {
        "severity": "major",
        "mechanism": "Valproic acid inhibits lamotrigine glucuronidation, approximately doubling lamotrigine levels.",
        "recommendation": "Reduce lamotrigine dose by 50% when combined with valproic acid. Titrate slowly.",
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
    # ── Additional common medications ──────────────────────────────────────────
    "acetaminophen": {
        "drug_class": "Analgesic / antipyretic",
        "standard_dose_range": "325mg-1000mg every 4-6 hours (max 4000mg/day; 2000mg/day with liver disease)",
        "common_side_effects": ["nausea", "rash (rare)"],
        "serious_side_effects": ["hepatotoxicity (overdose)", "acute liver failure", "Stevens-Johnson syndrome (rare)"],
        "contraindications": ["severe hepatic impairment", "active liver disease"],
    },
    "ibuprofen": {
        "drug_class": "Non-steroidal anti-inflammatory drug (NSAID)",
        "standard_dose_range": "200mg-800mg three times daily (max 3200mg/day)",
        "common_side_effects": ["dyspepsia", "nausea", "headache", "dizziness"],
        "serious_side_effects": ["GI bleeding", "renal impairment", "cardiovascular events", "peptic ulcer"],
        "contraindications": ["active GI bleeding", "severe renal impairment", "third trimester pregnancy", "CABG surgery"],
    },
    "naproxen": {
        "drug_class": "Non-steroidal anti-inflammatory drug (NSAID)",
        "standard_dose_range": "250mg-500mg twice daily (max 1250mg/day)",
        "common_side_effects": ["dyspepsia", "nausea", "headache", "drowsiness"],
        "serious_side_effects": ["GI bleeding", "renal impairment", "cardiovascular events"],
        "contraindications": ["active GI bleeding", "severe renal impairment", "third trimester pregnancy", "CABG surgery"],
    },
    "tramadol": {
        "drug_class": "Opioid analgesic (weak mu-agonist / SNRI)",
        "standard_dose_range": "50mg-100mg every 4-6 hours (max 400mg/day)",
        "common_side_effects": ["nausea", "dizziness", "constipation", "headache", "somnolence"],
        "serious_side_effects": ["seizures", "serotonin syndrome", "respiratory depression", "dependence"],
        "contraindications": ["concurrent MAOIs", "uncontrolled epilepsy", "severe respiratory depression"],
    },
    "morphine": {
        "drug_class": "Opioid analgesic (strong mu-agonist)",
        "standard_dose_range": "10mg-30mg every 4 hours (oral); titrated to effect in severe pain",
        "common_side_effects": ["constipation", "nausea", "sedation", "pruritus", "respiratory depression"],
        "serious_side_effects": ["severe respiratory depression", "hypotension", "dependence", "paralytic ileus"],
        "contraindications": ["severe respiratory depression", "acute or severe bronchial asthma", "paralytic ileus"],
    },
    "oxycodone": {
        "drug_class": "Opioid analgesic (strong mu-agonist)",
        "standard_dose_range": "5mg-15mg every 4-6 hours (immediate release); titrated to effect",
        "common_side_effects": ["constipation", "nausea", "somnolence", "dizziness", "pruritus"],
        "serious_side_effects": ["respiratory depression", "dependence", "hypotension"],
        "contraindications": ["significant respiratory depression", "acute or severe bronchial asthma", "paralytic ileus"],
    },
    "fluoxetine": {
        "drug_class": "Selective serotonin reuptake inhibitor (SSRI)",
        "standard_dose_range": "20mg-80mg once daily",
        "common_side_effects": ["nausea", "headache", "insomnia", "anxiety", "sexual dysfunction"],
        "serious_side_effects": ["serotonin syndrome", "suicidal ideation (young adults)", "hyponatraemia", "QT prolongation"],
        "contraindications": ["concurrent MAOI use", "concurrent pimozide or thioridazine"],
    },
    "escitalopram": {
        "drug_class": "Selective serotonin reuptake inhibitor (SSRI)",
        "standard_dose_range": "5mg-20mg once daily",
        "common_side_effects": ["nausea", "insomnia", "fatigue", "sexual dysfunction", "headache"],
        "serious_side_effects": ["serotonin syndrome", "QT prolongation", "suicidal ideation (young adults)", "hyponatraemia"],
        "contraindications": ["concurrent MAOI use", "concurrent pimozide", "QT prolongation"],
    },
    "venlafaxine": {
        "drug_class": "Serotonin-norepinephrine reuptake inhibitor (SNRI)",
        "standard_dose_range": "75mg-225mg once daily (extended release)",
        "common_side_effects": ["nausea", "headache", "dizziness", "insomnia", "sweating"],
        "serious_side_effects": ["serotonin syndrome", "hypertension", "suicidal ideation (young adults)", "discontinuation syndrome"],
        "contraindications": ["concurrent MAOI use", "uncontrolled hypertension"],
    },
    "duloxetine": {
        "drug_class": "Serotonin-norepinephrine reuptake inhibitor (SNRI)",
        "standard_dose_range": "30mg-120mg once daily",
        "common_side_effects": ["nausea", "dry mouth", "constipation", "fatigue", "dizziness"],
        "serious_side_effects": ["hepatotoxicity", "serotonin syndrome", "suicidal ideation", "hypertension"],
        "contraindications": ["concurrent MAOI use", "uncontrolled narrow-angle glaucoma", "severe hepatic impairment"],
    },
    "alprazolam": {
        "drug_class": "Benzodiazepine (anxiolytic)",
        "standard_dose_range": "0.25mg-0.5mg three times daily (max 4mg/day for anxiety)",
        "common_side_effects": ["drowsiness", "fatigue", "ataxia", "memory impairment"],
        "serious_side_effects": ["respiratory depression (with opioids)", "dependence", "paradoxical reactions", "withdrawal seizures"],
        "contraindications": ["acute narrow-angle glaucoma", "concurrent strong CYP3A4 inhibitors (ketoconazole, itraconazole)"],
    },
    "diazepam": {
        "drug_class": "Benzodiazepine (anxiolytic / anticonvulsant / muscle relaxant)",
        "standard_dose_range": "2mg-10mg two to four times daily",
        "common_side_effects": ["sedation", "fatigue", "ataxia", "muscle weakness"],
        "serious_side_effects": ["respiratory depression", "dependence", "paradoxical reactions"],
        "contraindications": ["acute narrow-angle glaucoma", "severe respiratory insufficiency", "myasthenia gravis"],
    },
    "lorazepam": {
        "drug_class": "Benzodiazepine (anxiolytic)",
        "standard_dose_range": "0.5mg-2mg two to three times daily",
        "common_side_effects": ["sedation", "dizziness", "weakness", "unsteadiness"],
        "serious_side_effects": ["respiratory depression", "dependence", "paradoxical agitation"],
        "contraindications": ["acute narrow-angle glaucoma", "severe respiratory insufficiency", "sleep apnoea"],
    },
    "zolpidem": {
        "drug_class": "Non-benzodiazepine hypnotic (Z-drug)",
        "standard_dose_range": "5mg-10mg at bedtime (5mg for women and elderly)",
        "common_side_effects": ["drowsiness", "dizziness", "headache", "diarrhoea"],
        "serious_side_effects": ["complex sleep behaviours (sleepwalking)", "respiratory depression", "dependence", "anaphylaxis"],
        "contraindications": ["severe respiratory insufficiency", "severe hepatic impairment", "myasthenia gravis"],
    },
    "rosuvastatin": {
        "drug_class": "HMG-CoA reductase inhibitor (statin)",
        "standard_dose_range": "5mg-40mg once daily",
        "common_side_effects": ["myalgia", "headache", "abdominal pain", "nausea"],
        "serious_side_effects": ["rhabdomyolysis", "hepatotoxicity", "new-onset diabetes"],
        "contraindications": ["active liver disease", "unexplained persistent transaminase elevation", "pregnancy"],
    },
    "pravastatin": {
        "drug_class": "HMG-CoA reductase inhibitor (statin)",
        "standard_dose_range": "10mg-80mg once daily",
        "common_side_effects": ["headache", "nausea", "myalgia", "fatigue"],
        "serious_side_effects": ["rhabdomyolysis", "hepatotoxicity"],
        "contraindications": ["active liver disease", "pregnancy"],
    },
    "spironolactone": {
        "drug_class": "Potassium-sparing diuretic / aldosterone antagonist",
        "standard_dose_range": "25mg-200mg daily",
        "common_side_effects": ["hyperkalaemia", "gynaecomastia", "breast tenderness", "dizziness"],
        "serious_side_effects": ["severe hyperkalaemia", "hyponatraemia", "metabolic acidosis"],
        "contraindications": ["hyperkalaemia", "Addison's disease", "anuria", "severe renal impairment"],
    },
    "digoxin": {
        "drug_class": "Cardiac glycoside",
        "standard_dose_range": "0.125mg-0.25mg once daily (target level 0.5-2.0 ng/mL)",
        "common_side_effects": ["nausea", "diarrhoea", "dizziness", "visual disturbances"],
        "serious_side_effects": ["cardiac arrhythmias", "heart block", "digoxin toxicity"],
        "contraindications": ["ventricular fibrillation", "hypertrophic obstructive cardiomyopathy"],
    },
    "amiodarone": {
        "drug_class": "Class III antiarrhythmic",
        "standard_dose_range": "200mg-400mg daily (after loading dose of 800-1600mg/day for 1-3 weeks)",
        "common_side_effects": ["nausea", "photosensitivity", "tremor", "corneal microdeposits"],
        "serious_side_effects": ["pulmonary toxicity", "thyroid dysfunction", "hepatotoxicity", "peripheral neuropathy", "QT prolongation"],
        "contraindications": ["sinus node disease", "second/third degree heart block", "severe thyroid dysfunction"],
    },
    "verapamil": {
        "drug_class": "Non-dihydropyridine calcium channel blocker",
        "standard_dose_range": "80mg-120mg three times daily or 120mg-480mg once daily (SR)",
        "common_side_effects": ["constipation", "dizziness", "headache", "peripheral oedema"],
        "serious_side_effects": ["bradycardia", "heart block", "heart failure exacerbation", "hypotension"],
        "contraindications": ["severe LV dysfunction", "second/third degree heart block", "sick sinus syndrome", "concurrent IV beta-blockers"],
    },
    "diltiazem": {
        "drug_class": "Non-dihydropyridine calcium channel blocker",
        "standard_dose_range": "120mg-360mg daily (extended release)",
        "common_side_effects": ["dizziness", "headache", "peripheral oedema", "bradycardia"],
        "serious_side_effects": ["severe bradycardia", "heart block", "heart failure"],
        "contraindications": ["severe LV dysfunction", "sick sinus syndrome", "second/third degree heart block"],
    },
    "empagliflozin": {
        "drug_class": "SGLT2 inhibitor (antidiabetic)",
        "standard_dose_range": "10mg-25mg once daily",
        "common_side_effects": ["genital mycotic infections", "urinary tract infections", "increased urination", "hypotension"],
        "serious_side_effects": ["diabetic ketoacidosis", "necrotising fasciitis of perineum", "acute kidney injury"],
        "contraindications": ["severe renal impairment (eGFR <20 mL/min for glycaemic control)", "dialysis"],
    },
    "dapagliflozin": {
        "drug_class": "SGLT2 inhibitor (antidiabetic)",
        "standard_dose_range": "5mg-10mg once daily",
        "common_side_effects": ["genital mycotic infections", "urinary tract infections", "back pain", "increased urination"],
        "serious_side_effects": ["diabetic ketoacidosis", "necrotising fasciitis of perineum", "volume depletion"],
        "contraindications": ["dialysis", "type 1 diabetes (for glycaemic control)"],
    },
    "sitagliptin": {
        "drug_class": "DPP-4 inhibitor (antidiabetic)",
        "standard_dose_range": "100mg once daily (50mg if eGFR 30-45; 25mg if eGFR <30)",
        "common_side_effects": ["headache", "nasopharyngitis", "upper respiratory infection"],
        "serious_side_effects": ["pancreatitis", "severe joint pain", "bullous pemphigoid"],
        "contraindications": ["history of pancreatitis with DPP-4 inhibitors"],
    },
    "glipizide": {
        "drug_class": "Sulfonylurea (antidiabetic)",
        "standard_dose_range": "2.5mg-20mg daily (max 40mg/day in divided doses)",
        "common_side_effects": ["hypoglycaemia", "weight gain", "nausea", "dizziness"],
        "serious_side_effects": ["severe hypoglycaemia", "haemolytic anaemia", "hepatotoxicity"],
        "contraindications": ["type 1 diabetes", "diabetic ketoacidosis", "severe hepatic impairment"],
    },
    "semaglutide": {
        "drug_class": "GLP-1 receptor agonist",
        "standard_dose_range": "0.25mg-2mg weekly (subcutaneous); 3mg-14mg once daily (oral)",
        "common_side_effects": ["nausea", "vomiting", "diarrhoea", "abdominal pain", "constipation"],
        "serious_side_effects": ["pancreatitis", "medullary thyroid carcinoma (animal studies)", "gallbladder disease", "acute kidney injury"],
        "contraindications": ["personal/family history of medullary thyroid carcinoma", "MEN 2 syndrome"],
    },
    "liraglutide": {
        "drug_class": "GLP-1 receptor agonist",
        "standard_dose_range": "0.6mg-1.8mg once daily (subcutaneous)",
        "common_side_effects": ["nausea", "vomiting", "diarrhoea", "headache"],
        "serious_side_effects": ["pancreatitis", "medullary thyroid carcinoma (animal studies)", "gallbladder disease"],
        "contraindications": ["personal/family history of medullary thyroid carcinoma", "MEN 2 syndrome"],
    },
    "rivaroxaban": {
        "drug_class": "Direct oral anticoagulant (Factor Xa inhibitor)",
        "standard_dose_range": "10mg-20mg once daily (indication-dependent)",
        "common_side_effects": ["bleeding", "bruising", "nausea", "anaemia"],
        "serious_side_effects": ["major haemorrhage", "spinal/epidural haematoma (with neuraxial procedures)"],
        "contraindications": ["active significant bleeding", "hepatic disease with coagulopathy", "concurrent strong CYP3A4/P-gp inhibitors"],
    },
    "apixaban": {
        "drug_class": "Direct oral anticoagulant (Factor Xa inhibitor)",
        "standard_dose_range": "2.5mg-5mg twice daily",
        "common_side_effects": ["bleeding", "bruising", "nausea"],
        "serious_side_effects": ["major haemorrhage", "spinal/epidural haematoma (with neuraxial procedures)"],
        "contraindications": ["active pathological bleeding", "severe hepatic disease"],
    },
    "dabigatran": {
        "drug_class": "Direct oral anticoagulant (direct thrombin inhibitor)",
        "standard_dose_range": "110mg-150mg twice daily",
        "common_side_effects": ["dyspepsia", "gastritis", "bleeding"],
        "serious_side_effects": ["major haemorrhage", "GI bleeding (higher vs warfarin)"],
        "contraindications": ["mechanical prosthetic heart valve", "active bleeding", "severe renal impairment (CrCl <30)"],
    },
    "enoxaparin": {
        "drug_class": "Low molecular weight heparin (anticoagulant)",
        "standard_dose_range": "40mg daily (prophylaxis); 1mg/kg twice daily (treatment)",
        "common_side_effects": ["injection site bruising", "bleeding", "thrombocytopenia"],
        "serious_side_effects": ["major bleeding", "heparin-induced thrombocytopenia", "spinal haematoma"],
        "contraindications": ["active major bleeding", "HIT", "severe thrombocytopenia"],
    },
    "tamsulosin": {
        "drug_class": "Alpha-1 adrenergic blocker (urological)",
        "standard_dose_range": "0.4mg-0.8mg once daily",
        "common_side_effects": ["dizziness", "orthostatic hypotension", "retrograde ejaculation", "rhinitis"],
        "serious_side_effects": ["intraoperative floppy iris syndrome", "priapism (rare)", "syncope"],
        "contraindications": ["concurrent strong CYP3A4 inhibitors (with 0.8mg dose)", "history of orthostatic hypotension"],
    },
    "sildenafil": {
        "drug_class": "PDE5 inhibitor",
        "standard_dose_range": "25mg-100mg as needed (max once daily)",
        "common_side_effects": ["headache", "flushing", "dyspepsia", "nasal congestion", "visual disturbance"],
        "serious_side_effects": ["priapism", "sudden hearing loss", "NAION (vision loss)", "severe hypotension"],
        "contraindications": ["concurrent nitrates", "severe hepatic impairment", "recent stroke or MI"],
    },
    "ondansetron": {
        "drug_class": "5-HT3 receptor antagonist (antiemetic)",
        "standard_dose_range": "4mg-8mg every 8 hours as needed",
        "common_side_effects": ["headache", "constipation", "fatigue"],
        "serious_side_effects": ["QT prolongation", "serotonin syndrome (with serotonergic drugs)", "anaphylaxis"],
        "contraindications": ["concurrent apomorphine", "congenital long QT syndrome"],
    },
    "metoclopramide": {
        "drug_class": "Dopamine antagonist (prokinetic / antiemetic)",
        "standard_dose_range": "5mg-10mg three times daily (max 12 weeks)",
        "common_side_effects": ["drowsiness", "restlessness", "fatigue", "diarrhoea"],
        "serious_side_effects": ["tardive dyskinesia", "neuroleptic malignant syndrome", "extrapyramidal symptoms"],
        "contraindications": ["GI obstruction or perforation", "phaeochromocytoma", "epilepsy", "concurrent drugs causing extrapyramidal reactions"],
    },
    "azithromycin": {
        "drug_class": "Macrolide antibiotic",
        "standard_dose_range": "250mg-500mg once daily (typically 3-5 day course)",
        "common_side_effects": ["diarrhoea", "nausea", "abdominal pain", "vomiting"],
        "serious_side_effects": ["QT prolongation", "hepatotoxicity", "C. difficile colitis", "hearing loss"],
        "contraindications": ["history of cholestatic jaundice with azithromycin", "hepatic impairment from prior azithromycin"],
    },
    "doxycycline": {
        "drug_class": "Tetracycline antibiotic",
        "standard_dose_range": "100mg twice daily or 200mg once daily",
        "common_side_effects": ["nausea", "photosensitivity", "oesophageal irritation", "diarrhoea"],
        "serious_side_effects": ["oesophageal ulceration", "intracranial hypertension", "hepatotoxicity", "tooth discolouration (children)"],
        "contraindications": ["pregnancy", "children under 8 years", "severe hepatic impairment"],
    },
    "metronidazole": {
        "drug_class": "Nitroimidazole antibiotic / antiprotozoal",
        "standard_dose_range": "250mg-500mg three times daily (7-14 days typical)",
        "common_side_effects": ["nausea", "metallic taste", "headache", "dark urine"],
        "serious_side_effects": ["peripheral neuropathy", "seizures", "disulfiram-like reaction with alcohol"],
        "contraindications": ["first trimester pregnancy (relative)", "concurrent alcohol use"],
    },
    "fluconazole": {
        "drug_class": "Triazole antifungal",
        "standard_dose_range": "50mg-400mg once daily (indication-dependent)",
        "common_side_effects": ["nausea", "headache", "abdominal pain", "diarrhoea"],
        "serious_side_effects": ["hepatotoxicity", "QT prolongation", "Stevens-Johnson syndrome", "adrenal insufficiency"],
        "contraindications": ["concurrent terfenadine or cisapride (QT risk)", "hypersensitivity to azoles"],
    },
    "trimethoprim": {
        "drug_class": "Dihydrofolate reductase inhibitor (antibiotic)",
        "standard_dose_range": "100mg-200mg twice daily (often combined with sulfamethoxazole)",
        "common_side_effects": ["nausea", "rash", "pruritus", "hyperkalaemia"],
        "serious_side_effects": ["megaloblastic anaemia", "pancytopenia", "Stevens-Johnson syndrome", "hyperkalaemia"],
        "contraindications": ["megaloblastic anaemia due to folate deficiency", "severe renal impairment (for TMP/SMX)"],
    },
    "albuterol": {
        "drug_class": "Short-acting beta-2 agonist (bronchodilator)",
        "standard_dose_range": "2 puffs (90mcg/puff) every 4-6 hours as needed; 2.5mg nebulised",
        "common_side_effects": ["tremor", "tachycardia", "headache", "nervousness"],
        "serious_side_effects": ["paradoxical bronchospasm", "hypokalaemia", "cardiac arrhythmias"],
        "contraindications": ["hypersensitivity to albuterol"],
    },
    "montelukast": {
        "drug_class": "Leukotriene receptor antagonist",
        "standard_dose_range": "10mg once daily at bedtime (adults)",
        "common_side_effects": ["headache", "abdominal pain", "cough"],
        "serious_side_effects": ["neuropsychiatric events (depression, suicidal ideation — FDA boxed warning)", "Churg-Strauss syndrome"],
        "contraindications": ["hypersensitivity to montelukast"],
    },
    "tiotropium": {
        "drug_class": "Long-acting muscarinic antagonist (LAMA) — inhaled anticholinergic",
        "standard_dose_range": "18mcg (capsule) or 2.5mcg (Respimat) once daily by inhalation",
        "common_side_effects": ["dry mouth", "pharyngitis", "upper respiratory tract infection"],
        "serious_side_effects": ["paradoxical bronchospasm", "urinary retention", "angle-closure glaucoma"],
        "contraindications": ["hypersensitivity to tiotropium or atropine derivatives"],
    },
    "methotrexate": {
        "drug_class": "Antimetabolite / disease-modifying antirheumatic drug (DMARD)",
        "standard_dose_range": "7.5mg-25mg once weekly (RA/psoriasis); variable for oncology",
        "common_side_effects": ["nausea", "fatigue", "mouth sores", "abdominal discomfort"],
        "serious_side_effects": ["myelosuppression", "hepatotoxicity", "pulmonary toxicity", "nephrotoxicity"],
        "contraindications": ["pregnancy", "breastfeeding", "alcoholism", "immunodeficiency", "pre-existing blood dyscrasias"],
    },
    "lithium": {
        "drug_class": "Mood stabiliser",
        "standard_dose_range": "300mg-600mg two to three times daily (target level 0.6-1.2 mEq/L)",
        "common_side_effects": ["tremor", "thirst", "polyuria", "weight gain", "GI upset"],
        "serious_side_effects": ["lithium toxicity", "nephrogenic diabetes insipidus", "hypothyroidism", "cardiac arrhythmias"],
        "contraindications": ["severe renal impairment", "Brugada syndrome", "Addison's disease"],
    },
    "carbamazepine": {
        "drug_class": "Anticonvulsant / mood stabiliser",
        "standard_dose_range": "200mg-1200mg daily in divided doses",
        "common_side_effects": ["dizziness", "drowsiness", "nausea", "ataxia", "diplopia"],
        "serious_side_effects": ["Stevens-Johnson syndrome / TEN", "aplastic anaemia", "agranulocytosis", "hyponatraemia", "hepatotoxicity"],
        "contraindications": ["bone marrow depression", "concurrent MAOIs", "AV conduction abnormalities", "HLA-B*1502 positive (SJS risk)"],
    },
    "valproic acid": {
        "drug_class": "Anticonvulsant / mood stabiliser",
        "standard_dose_range": "250mg-1000mg two to three times daily (target level 50-100 mcg/mL)",
        "common_side_effects": ["nausea", "tremor", "weight gain", "alopecia", "drowsiness"],
        "serious_side_effects": ["hepatotoxicity", "pancreatitis", "thrombocytopenia", "teratogenicity (neural tube defects)"],
        "contraindications": ["hepatic disease", "urea cycle disorders", "pregnancy", "mitochondrial disorders (POLG mutations)"],
    },
    "lamotrigine": {
        "drug_class": "Anticonvulsant / mood stabiliser",
        "standard_dose_range": "25mg-400mg daily (must titrate slowly; lower with valproate)",
        "common_side_effects": ["headache", "dizziness", "diplopia", "nausea", "rash"],
        "serious_side_effects": ["Stevens-Johnson syndrome / TEN", "aseptic meningitis", "haemophagocytic lymphohistiocytosis"],
        "contraindications": ["hypersensitivity to lamotrigine"],
    },
    "phenytoin": {
        "drug_class": "Anticonvulsant (hydantoin)",
        "standard_dose_range": "100mg two to three times daily (target level 10-20 mcg/mL)",
        "common_side_effects": ["dizziness", "nystagmus", "gingival hyperplasia", "hirsutism", "ataxia"],
        "serious_side_effects": ["Stevens-Johnson syndrome / TEN", "hepatotoxicity", "megaloblastic anaemia", "purple glove syndrome (IV)"],
        "contraindications": ["sinus bradycardia", "sinoatrial block", "second/third degree heart block", "Adams-Stokes syndrome"],
    },
}


def _normalize_drug_name(name: str) -> str:
    """Normalize drug name for lookup."""
    return name.strip().lower().replace("-", " ").replace("_", " ")


def _llm_drug_interaction(drug_a: str, drug_b: str) -> str:
    """Use the LLM to analyse a drug-drug interaction not in the local table."""
    logger.info("llm_drug_interaction drug_a=%s drug_b=%s", drug_a, drug_b)
    try:
        response = litellm.completion(
            model=_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical pharmacologist. Analyse the potential interaction "
                        "between two medications. Respond ONLY with a JSON object containing: "
                        "severity (none/minor/moderate/major), mechanism (pharmacological explanation), "
                        "recommendation (clinical action). Be concise and evidence-based. "
                        "If there is genuinely no interaction, set severity to 'none'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyse the drug-drug interaction between {drug_a} and {drug_b}.",
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON from the LLM response (handle markdown code fences)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        parsed = json.loads(content)
        return json.dumps({
            "status": "interaction_found" if parsed.get("severity", "none") != "none" else "no_interaction",
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": parsed.get("severity", "unknown"),
            "mechanism": parsed.get("mechanism", ""),
            "recommendation": parsed.get("recommendation", ""),
            "source": "ai_analysis",
            "note": "Analysis generated by AI — verify with a pharmacist or drug interaction database for clinical decisions.",
        })
    except Exception as e:
        logger.warning("llm_drug_interaction_failed error=%s", e)
        return json.dumps({
            "status": "analysis_unavailable",
            "drug_a": drug_a,
            "drug_b": drug_b,
            "note": "Neither the built-in reference table nor AI analysis could evaluate this pair. Consult a pharmacist.",
        })


def _llm_medication_info(drug_name: str) -> str:
    """Use the LLM to provide medication information not in the local table."""
    logger.info("llm_medication_info drug=%s", drug_name)
    try:
        response = litellm.completion(
            model=_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical pharmacologist. Provide medication information. "
                        "Respond ONLY with a JSON object containing: drug_class, "
                        "standard_dose_range, common_side_effects (array), "
                        "serious_side_effects (array), contraindications (array). "
                        "Be concise, accurate, and evidence-based."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Provide clinical information for the medication: {drug_name}.",
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        parsed = json.loads(content)
        return json.dumps({
            "status": "found",
            "drug_name": drug_name,
            "source": "ai_analysis",
            "note": "Information generated by AI — verify with official prescribing information for clinical decisions.",
            **parsed,
        })
    except Exception as e:
        logger.warning("llm_medication_info_failed error=%s", e)
        return json.dumps({
            "status": "not_found",
            "drug_name": drug_name,
            "note": "Medication not found in built-in reference table and AI analysis is unavailable.",
        })


@crewai_tool("Check Drug Interactions")
def check_drug_interactions(drug_a: str, drug_b: str) -> str:
    """
    Check for drug-drug interactions between two medications.
    First checks a built-in verified reference table of 30+ common interaction
    pairs, then falls back to AI-powered pharmacological analysis for any
    drug pair not in the table.

    Args:
        drug_a: First medication name.
        drug_b: Second medication name.
    """
    a = _normalize_drug_name(drug_a)
    b = _normalize_drug_name(drug_b)

    logger.info("tool_check_drug_interactions drug_a=%s drug_b=%s", a, b)

    # 1. Exact match in verified reference table
    interaction = _DRUG_INTERACTIONS.get((a, b)) or _DRUG_INTERACTIONS.get((b, a))

    if interaction:
        return json.dumps({
            "status": "interaction_found",
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": interaction["severity"],
            "mechanism": interaction["mechanism"],
            "recommendation": interaction["recommendation"],
            "source": "verified_reference",
        })

    # 2. Partial match in verified reference table (min 4-char names to avoid false positives)
    if len(a) >= 4 and len(b) >= 4:
        for (da, db), info in _DRUG_INTERACTIONS.items():
            pairs_to_check = [(a, da, b, db), (a, db, b, da)]
            for (x, xref, y, yref) in pairs_to_check:
                if (x in xref or xref in x) and (y in yref or yref in y):
                    return json.dumps({
                        "status": "interaction_found",
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        "matched_pair": f"{da} + {db}",
                        "severity": info["severity"],
                        "mechanism": info["mechanism"],
                        "recommendation": info["recommendation"],
                        "source": "verified_reference",
                        "note": "Matched via partial name matching.",
                    })

    # 3. Fall back to AI-powered analysis
    return _llm_drug_interaction(drug_a, drug_b)


@crewai_tool("Get Medication Info")
def get_medication_info(drug_name: str) -> str:
    """
    Get medication information including drug class, dosage range, side effects,
    and contraindications. First checks a built-in verified reference table of
    20+ common medications, then falls back to AI-powered analysis for any
    medication not in the table.

    Args:
        drug_name: Name of the medication to look up.
    """
    name = _normalize_drug_name(drug_name)
    logger.info("tool_get_medication_info drug=%s", name)

    # 1. Exact match
    info = _MEDICATION_INFO.get(name)
    if info:
        return json.dumps({
            "status": "found",
            "drug_name": drug_name,
            "source": "verified_reference",
            **info,
        })

    # 2. Partial match (min 4 chars to avoid false positives)
    if len(name) >= 4:
        for key, med_info in _MEDICATION_INFO.items():
            if name in key or key in name:
                return json.dumps({
                    "status": "found",
                    "drug_name": drug_name,
                    "matched_name": key,
                    "source": "verified_reference",
                    "note": "Matched via partial name matching.",
                    **med_info,
                })

    # 3. Fall back to AI-powered analysis
    return _llm_medication_info(drug_name)
