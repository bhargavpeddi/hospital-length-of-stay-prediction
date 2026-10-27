"""Step 0: download NY SPARCS 2022 inpatient discharges (de-identified) from health.data.ny.gov.
Dataset: Hospital Inpatient Discharges (SPARCS De-Identified): 2022, id 5dtw-tffi (about 2.1M rows).
Writes data/sparcs_2022_los.csv.gz with only the columns this project uses.
"""
import gzip, pathlib, urllib.request

COLS = ("hospital_service_area,permanent_facility_id,facility_name,age_group,gender,race,ethnicity,length_of_stay,"
        "type_of_admission,patient_disposition,ccsr_diagnosis_code,ccsr_diagnosis_description,apr_drg_code,"
        "apr_drg_description,apr_mdc_code,apr_mdc_description,apr_severity_of_illness_code,apr_risk_of_mortality,"
        "apr_medical_surgical,payment_typology_1,emergency_department_indicator,total_charges,total_costs")
URL = "https://health.data.ny.gov/resource/5dtw-tffi.csv?$select={c}&$order=:id&$limit={l}&$offset={o}"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "sparcs_2022_los.csv.gz"
OUT.parent.mkdir(exist_ok=True)
LIMIT, offset, first = 50000, 0, True
with gzip.open(OUT, "wt", encoding="utf-8") as f:
    while True:
        txt = urllib.request.urlopen(URL.format(c=COLS, l=LIMIT, o=offset), timeout=300).read().decode("utf-8")
        lines = txt.splitlines(keepends=True)
        body = lines if first else lines[1:]
        if len(lines) <= 1:
            break
        f.writelines(body); first = False; offset += LIMIT
        print(f"downloaded {offset:,} rows")
print("saved", OUT)
