"""Step 1: load NY SPARCS 2022 inpatient discharges, validate every row, keep clean rows.

Input : data/sparcs_2022_los.csv.gz  (health.data.ny.gov dataset 5dtw-tffi, selected columns)
Output: data/clean.parquet, outputs/tables/rejected_rows_summary.csv, outputs/tables/validation_log.txt
"""
import pandas as pd, numpy as np, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "sparcs_2022_los.csv.gz"
T = ROOT / "outputs" / "tables"; T.mkdir(parents=True, exist_ok=True)

t0 = time.time()
df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
n_raw = len(df)
log = [f"Raw rows loaded: {n_raw:,}"]

# Length of stay: SPARCS caps very long stays as "120 +"
los_raw = df["length_of_stay"].str.strip()
df["los_capped_flag"] = los_raw.eq("120 +")
df["length_of_stay"] = pd.to_numeric(los_raw.str.replace(" +", "", regex=False), errors="coerce")
for c in ["total_charges", "total_costs"]:
    df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")

checks = {
    "missing_or_bad_length_of_stay": df["length_of_stay"].isna() | (df["length_of_stay"] < 1),
    "missing_age_group": df["age_group"].eq(""),
    "unknown_type_of_admission": df["type_of_admission"].isin(["", "Not Available"]),
    "missing_diagnosis_or_drg": df["ccsr_diagnosis_code"].eq("") | df["apr_drg_code"].eq(""),
    "missing_severity": ~df["apr_severity_of_illness_code"].isin(["1", "2", "3", "4"]),
    "missing_facility": df["permanent_facility_id"].eq(""),
    "unknown_gender": ~df["gender"].isin(["F", "M"]),
    "bad_charges": df["total_charges"].isna() | (df["total_charges"] <= 0),
}
bad_any = np.zeros(len(df), dtype=bool)
rows = []
for name, m in checks.items():
    rows.append((name, int(m.sum())))
    bad_any |= m.values
pd.DataFrame(rows, columns=["check", "rows_failing"]).to_csv(T / "rejected_rows_summary.csv", index=False)

clean = df.loc[~bad_any].copy()
log.append(f"Rows failing at least one check: {int(bad_any.sum()):,} ({bad_any.mean():.2%})")
log.append(f"Clean rows kept: {len(clean):,}")
log.append(f"Stays capped at '120 +' days: {int(clean['los_capped_flag'].sum()):,}")
for name, n in rows:
    log.append(f"  {name}: {n:,}")

clean["apr_severity_of_illness_code"] = clean["apr_severity_of_illness_code"].astype(int)
for c in clean.columns:
    if clean[c].dtype == object:
        clean[c] = clean[c].astype("category")
clean.to_parquet(ROOT / "data" / "clean.parquet", index=False)
log.append(f"Done in {time.time()-t0:.0f}s")
(T / "validation_log.txt").write_text("\n".join(log) + "\n")
print("\n".join(log))
