"""Step 2: exploratory analysis of length of stay (LOS). Writes Power BI-ready tables and charts."""
import pandas as pd, numpy as np, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"; C = ROOT / "outputs" / "charts"
C.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(ROOT / "data" / "clean.parquet")
df["long_stay"] = df["length_of_stay"] > 7
BLUE, GREY = "#1F4E79", "#9AA5B1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def summ(g):
    return g.agg(stays=("length_of_stay", "size"), mean_los=("length_of_stay", "mean"),
                 median_los=("length_of_stay", "median"), long_stay_rate=("long_stay", "mean"),
                 bed_days=("length_of_stay", "sum")).reset_index()

overall = {
    "stays": len(df), "mean_los": df.length_of_stay.mean(), "median_los": df.length_of_stay.median(),
    "p90_los": df.length_of_stay.quantile(.9), "long_stay_rate": df.long_stay.mean(),
    "long_stay_share_of_bed_days": df.loc[df.long_stay, "length_of_stay"].sum() / df.length_of_stay.sum(),
}
pd.Series(overall).to_csv(T / "overall_summary.csv", header=["value"])

SEV = {1: "1 Minor", 2: "2 Moderate", 3: "3 Major", 4: "4 Extreme"}
by_sev = summ(df.groupby("apr_severity_of_illness_code", observed=True)); by_sev["severity"] = by_sev.apr_severity_of_illness_code.map(SEV)
by_sev.to_csv(T / "los_by_severity.csv", index=False)
by_adm = summ(df.groupby("type_of_admission", observed=True)).sort_values("stays", ascending=False); by_adm.to_csv(T / "los_by_admission_type.csv", index=False)
by_age = summ(df.groupby("age_group", observed=True)); by_age.to_csv(T / "los_by_age_group.csv", index=False)
by_mdc = summ(df.groupby("apr_mdc_description", observed=True)).sort_values("bed_days", ascending=False); by_mdc.to_csv(T / "los_by_mdc.csv", index=False)
by_drg = summ(df.groupby(["apr_drg_code", "apr_drg_description"], observed=True)).query("stays >= 1000").sort_values("bed_days", ascending=False)
by_drg.to_csv(T / "los_by_drg.csv", index=False)
by_hosp = summ(df.groupby(["permanent_facility_id", "facility_name", "hospital_service_area"], observed=True)).query("stays >= 500")
# severity-adjusted: expected LOS per stay = mean LOS of its DRG x severity cell; O/E ratio per hospital
cell = df.groupby(["apr_drg_code", "apr_severity_of_illness_code"], observed=True)["length_of_stay"].transform("mean")
df["expected_los"] = cell
oe = df.groupby("permanent_facility_id", observed=True).agg(obs=("length_of_stay", "sum"), exp=("expected_los", "sum")).reset_index()
oe["los_o_e_ratio"] = oe.obs / oe.exp
by_hosp = by_hosp.merge(oe[["permanent_facility_id", "los_o_e_ratio"]], on="permanent_facility_id")
by_hosp.sort_values("bed_days", ascending=False).to_csv(T / "los_by_hospital.csv", index=False)
summ(df.groupby("payment_typology_1", observed=True)).sort_values("stays", ascending=False).to_csv(T / "los_by_payer.csv", index=False)

# --- charts ---
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(df.length_of_stay.clip(upper=30), bins=30, color=BLUE)
ax.set_xlabel("Length of stay (days, 30+ grouped)"); ax.set_ylabel("Hospital stays")
ax.set_title("Most stays are short; a long tail drives bed use")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x/1000:.0f}K"))
fig.tight_layout(); fig.savefig(C / "01_los_distribution.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
s = by_sev.sort_values("apr_severity_of_illness_code")
lab = s.severity
ax.bar(lab, s.mean_los, color=BLUE)
for i, v in enumerate(s.mean_los): ax.text(i, v + .2, f"{v:.1f}", ha="center")
ax.set_ylabel("Mean LOS (days)"); ax.set_title("Severity of illness is the strongest single driver")
fig.tight_layout(); fig.savefig(C / "02_los_by_severity.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.8))
a = by_age.copy(); order = ["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"]
a = a.set_index("age_group").reindex([o for o in order if o in set(a.age_group)])
ax.bar(a.index, a.mean_los, color=BLUE); ax2 = ax.twinx()
ax2.plot(a.index, a.long_stay_rate * 100, color="#C0504D", marker="o"); ax2.set_ylabel("Long stays (>7 days), %")
ax.set_ylabel("Mean LOS (days)"); ax.set_title("Older patients stay longer and more often exceed a week")
ax2.spines["top"].set_visible(False)
fig.tight_layout(); fig.savefig(C / "03_los_by_age.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
d = by_drg.head(12).iloc[::-1]
ax.barh(d.apr_drg_description.str.slice(0, 45), d.bed_days / 1000, color=BLUE)
ax.set_xlabel("Bed days (thousands)"); ax.set_title("Top 12 APR-DRGs by total bed days", loc="left")
fig.tight_layout(); fig.savefig(C / "04_top_drg_bed_days.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.8))
h = by_hosp.query("stays >= 2000")
ax.scatter(h.stays / 1000, h.los_o_e_ratio, s=12, color=BLUE, alpha=.7)
ax.axhline(1, color=GREY, ls="--"); ax.set_xlabel("Hospital volume (thousand stays)")
ax.set_ylabel("Observed / expected LOS"); ax.set_title("Same case mix, different stays: hospital LOS O/E ratio")
fig.tight_layout(); fig.savefig(C / "05_hospital_oe_ratio.png", dpi=150); plt.close(fig)

hs = by_hosp.query("stays >= 2000")
lines = [f"{k}: {v:,.3f}" if isinstance(v, float) else f"{k}: {v:,}" for k, v in overall.items()]
lines += ["", "Severity:"] + [f"  {r.apr_severity_of_illness_code}: mean {r.mean_los:.2f}, long {r.long_stay_rate:.1%}, n {r.stays:,}" for r in s.itertuples()]
lines += ["", "Age:"] + [f"  {i}: mean {r.mean_los:.2f}, long {r.long_stay_rate:.1%}" for i, r in a.iterrows()]
lines += ["", "Admission type:"] + [f"  {r.type_of_admission}: n {r.stays:,}, mean {r.mean_los:.2f}" for r in by_adm.itertuples()]
lines += ["", "Top DRG by bed days:"] + [f"  {r.apr_drg_description[:50]}: {r.bed_days:,} days, mean {r.mean_los:.1f}" for r in by_drg.head(5).itertuples()]
lines += ["", f"Hospitals with >=2000 stays: {len(hs)}; O/E p10 {hs.los_o_e_ratio.quantile(.1):.2f}, p90 {hs.los_o_e_ratio.quantile(.9):.2f}",
          f"Hospitals with O/E > 1.10: {(hs.los_o_e_ratio>1.1).sum()}; excess bed days at those vs expected: {((hs.los_o_e_ratio-1).clip(lower=0)*hs.bed_days/hs.los_o_e_ratio).sum():,.0f}"]
(T / "eda_findings.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
