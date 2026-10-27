"""Step 3: predict length of stay with XGBoost.
  (a) regression: LOS in days      (b) classification: long stay (> 7 days)
Leakage control: discharge-time fields (patient disposition, charges, costs) are NOT used as features.
Baseline = mean LOS (or long-stay rate) of the stay's APR-DRG x severity cell, learned on the training set only.
"""
import pandas as pd, numpy as np, pathlib, json, time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"; C = ROOT / "outputs" / "charts"; M = ROOT / "outputs" / "models"; M.mkdir(exist_ok=True, parents=True)
BLUE = "#1F4E79"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_parquet(ROOT / "data" / "clean.parquet")
FEATS = ["age_group", "gender", "race", "ethnicity", "type_of_admission", "hospital_service_area", "permanent_facility_id",
         "ccsr_diagnosis_code", "apr_drg_code", "apr_mdc_code", "apr_severity_of_illness_code", "apr_risk_of_mortality",
         "apr_medical_surgical", "payment_typology_1", "emergency_department_indicator"]
X = df[FEATS].copy()
for c in FEATS:
    if c != "apr_severity_of_illness_code":
        X[c] = X[c].astype(str).astype("category")
y_days = df["length_of_stay"].astype(float).values
y_long = (y_days > 7).astype(int)

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=y_long)
idx_tr, idx_va = train_test_split(idx_tr, test_size=0.1, random_state=42, stratify=y_long[idx_tr])
Xtr, Xva, Xte = X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te]
res = {"rows_train": len(idx_tr), "rows_valid": len(idx_va), "rows_test": len(idx_te)}

# ---- baselines from training data only
cell = pd.DataFrame({"drg": df.apr_drg_code.astype(str).values, "sev": df.apr_severity_of_illness_code.values, "los": y_days, "long": y_long})
tr = cell.iloc[idx_tr]
cm = tr.groupby(["drg", "sev"]).agg(m=("los", "mean"), r=("long", "mean")).reset_index()
te = cell.iloc[idx_te].merge(cm, on=["drg", "sev"], how="left")
te["m"] = te["m"].fillna(tr.los.mean()); te["r"] = te["r"].fillna(tr.long.mean())
yte_d, yte_l = y_days[idx_te], y_long[idx_te]

def reg_metrics(p):
    return {"MAE_days": mean_absolute_error(yte_d, p), "RMSE_days": float(np.sqrt(mean_squared_error(yte_d, p))), "R2": r2_score(yte_d, p)}
res["reg_baseline_global_mean"] = reg_metrics(np.full(len(yte_d), y_days[idx_tr].mean()))
res["reg_baseline_drg_x_severity"] = reg_metrics(te["m"].values)

P = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, max_depth=8, learning_rate=0.08,
         subsample=0.8, colsample_bytree=0.8, min_child_weight=20, max_bin=256, nthread=2)

t0 = time.time()
reg = xgb.XGBRegressor(n_estimators=1500, objective="reg:squarederror", early_stopping_rounds=40, eval_metric="mae", **P)
reg.fit(Xtr, np.log1p(y_days[idx_tr]), eval_set=[(Xva, np.log1p(y_days[idx_va]))], verbose=False)
p_log = reg.predict(Xte)
# log-target predicts the median-ish; scale to the mean with a smearing factor fitted on validation
resid = np.log1p(y_days[idx_va]) - reg.predict(Xva); smear = np.mean(np.exp(resid))
p_days = np.clip(np.expm1(p_log) * smear, 1, 120)
p_med = np.clip(np.expm1(p_log), 1, 120)
res["reg_xgboost"] = reg_metrics(p_days); res["reg_xgboost"]["MAE_days_median_pred"] = mean_absolute_error(yte_d, p_med)
res["reg_xgboost"]["best_iteration"] = int(reg.best_iteration); res["reg_xgboost"]["train_seconds"] = round(time.time() - t0)
reg.save_model(M / "los_regressor.json")

t0 = time.time()
clf = xgb.XGBClassifier(n_estimators=1500, objective="binary:logistic", early_stopping_rounds=40, eval_metric="auc", **P)
clf.fit(Xtr, y_long[idx_tr], eval_set=[(Xva, y_long[idx_va])], verbose=False)
p_cls = clf.predict_proba(Xte)[:, 1]
clf.save_model(M / "long_stay_classifier.json")

def cls_metrics(p):
    k = int(len(p) * 0.2); top = np.argsort(-p)[:k]
    return {"ROC_AUC": roc_auc_score(yte_l, p), "PR_AUC": average_precision_score(yte_l, p),
            "recall_top20pct": yte_l[top].sum() / yte_l.sum(), "precision_top20pct": yte_l[top].mean()}
res["cls_base_rate"] = float(yte_l.mean())
res["cls_baseline_drg_x_severity"] = cls_metrics(te["r"].values)
res["cls_xgboost"] = cls_metrics(p_cls); res["cls_xgboost"]["best_iteration"] = int(clf.best_iteration); res["cls_xgboost"]["train_seconds"] = round(time.time() - t0)

# ---- bed-day view: share of long-stay bed days in the top 20% flagged stays
k = int(len(p_cls) * 0.2); top = np.argsort(-p_cls)[:k]
res["cls_xgboost"]["share_of_bed_days_in_top20pct"] = float(yte_d[top].sum() / yte_d.sum())

# ---- SHAP-style contributions (XGBoost native TreeSHAP) on a test sample
rs = np.random.RandomState(7); samp = rs.choice(len(idx_te), 20000, replace=False)
dm = xgb.DMatrix(Xte.iloc[samp], enable_categorical=True)
contrib = clf.get_booster().predict(dm, pred_contribs=True)[:, :-1]
imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(contrib).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp.to_csv(T / "shap_importance_long_stay.csv", index=False)
contrib_r = reg.get_booster().predict(dm, pred_contribs=True)[:, :-1]
imp_r = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(contrib_r).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp_r.to_csv(T / "shap_importance_los_regression.csv", index=False)

NICE = {"apr_drg_code": "Diagnosis group (APR-DRG)", "apr_severity_of_illness_code": "Severity of illness", "ccsr_diagnosis_code": "Principal diagnosis (CCSR)",
        "permanent_facility_id": "Hospital", "apr_risk_of_mortality": "Risk of mortality", "age_group": "Age group", "type_of_admission": "Admission type",
        "payment_typology_1": "Payer", "apr_mdc_code": "Body system (MDC)", "hospital_service_area": "Region", "emergency_department_indicator": "Came through ED",
        "apr_medical_surgical": "Medical vs surgical", "race": "Race", "ethnicity": "Ethnicity", "gender": "Gender"}
fig, ax = plt.subplots(figsize=(7, 4.2))
d = imp.head(10).iloc[::-1]
ax.barh(d.feature.map(NICE), d.mean_abs_shap, color=BLUE)
ax.set_xlabel("Mean |SHAP| (log-odds of a stay > 7 days)"); ax.set_title("What drives a long stay (SHAP, 20K test stays)")
fig.tight_layout(); fig.savefig(C / "06_shap_long_stay.png", dpi=150); plt.close(fig)

# ---- calibration-style chart: predicted risk decile vs actual long-stay rate
dec = pd.qcut(p_cls, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"decile": dec + 1, "pred": p_cls, "actual": yte_l, "los": yte_d}).groupby("decile").agg(pred=("pred", "mean"), actual=("actual", "mean"), mean_los=("los", "mean")).reset_index()
cal.to_csv(T / "risk_deciles.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(cal.decile, cal.actual * 100, color=BLUE, label="Actual long-stay rate")
ax.plot(cal.decile, cal.pred * 100, color="#C0504D", marker="o", label="Predicted")
ax.set_xlabel("Risk decile (1 = lowest predicted risk)"); ax.set_ylabel("% of stays > 7 days"); ax.set_xticks(cal.decile)
ax.set_title("Long-stay model ranks risk well across deciles"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(C / "07_risk_deciles.png", dpi=150); plt.close(fig)

# ---- error by LOS band for the regression model
band = pd.cut(yte_d, [0, 3, 7, 14, 30, 121], labels=["1-3", "4-7", "8-14", "15-30", "31+"])
eb = pd.DataFrame({"band": band, "abs_err": np.abs(p_days - yte_d), "base_err": np.abs(te["m"].values - yte_d)}).groupby("band", observed=True).agg(
    stays=("abs_err", "size"), model_MAE=("abs_err", "mean"), baseline_MAE=("base_err", "mean")).reset_index()
eb.to_csv(T / "regression_error_by_los_band.csv", index=False)

json.dump(res, open(T / "model_results.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float)); print(imp.head(8).to_string(index=False)); print(eb.to_string(index=False))
