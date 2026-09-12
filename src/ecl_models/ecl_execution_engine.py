from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ECLExecutionEngine:
    def __init__(
        self,
        portfolio_path,
        macro_scenarios_path,
        config_path="config/config.yaml",
        root=".",
        experiment_name="IFRS9_ECL_Modeling",
    ):
        self.root = Path(root)
        self.config = _load_yaml(str(self.root / config_path))
        self.portfolio = pd.read_csv(self.root / portfolio_path, low_memory=False)
        self.scenarios = pd.read_csv(self.root / macro_scenarios_path)
        self.weights = self.config.get("scenario_weights", {"baseline": 1.0})
        self.ecl_cfg = self.config["ecl"]
        self.macro_lags = list(self.config["model_features"]["macro_lags"])
        self.pd_features = self.config["model_features"]["pd_financials"] + self.macro_lags
        gdp_lags = [m for m in self.macro_lags if "gdp" in m.lower()]
        self.lgd_features = self.config["model_features"]["lgd_financials"] + (gdp_lags or self.macro_lags[:1])
        self.run_id = "local_artifacts"
        self._load_models(experiment_name)

    def _load_models(self, experiment_name: str) -> None:
        pd_path = self.root / self.config["paths"]["pd_model"]
        lgd_path = self.root / self.config["paths"]["lgd_model"]
        if pd_path.exists() and lgd_path.exists():
            pd_bundle = joblib.load(pd_path)
            self.pd_model = pd_bundle["horizon_1y"] if isinstance(pd_bundle, dict) else pd_bundle
            self.lgd_model = joblib.load(lgd_path)
            return
        try:
            import mlflow.sklearn
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            exp = client.get_experiment_by_name(experiment_name)
            if not exp:
                raise ValueError(f"MLflow experiment {experiment_name} not found and no local models/")
            runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
            self.run_id = runs[0].info.run_id
            self.pd_model = mlflow.sklearn.load_model(f"runs:/{self.run_id}/pd_model_horizon_1y")
            self.lgd_model = {
                "classifier": mlflow.sklearn.load_model(f"runs:/{self.run_id}/lgd_classifier"),
                "regressor": mlflow.sklearn.load_model(f"runs:/{self.run_id}/lgd_regressor"),
            }
        except Exception as e:
            raise FileNotFoundError(f"Could not load PD/LGD models: {e}") from e

    def _map_macro_features(self, df: pd.DataFrame, macro_row: pd.Series) -> pd.DataFrame:
        df = df.copy()
        for col in self.macro_lags:
            if col in macro_row.index and pd.notna(macro_row[col]):
                df[col] = macro_row[col]
                continue
            base = col.rsplit("_lag_", 1)[0]
            if base in macro_row.index:
                df[col] = macro_row[base]
                continue
            matched = 0.0
            for sc_col in macro_row.index:
                if str(base) in str(sc_col):
                    matched = macro_row[sc_col]
                    break
            df[col] = matched
        return df

    def _apply_pd_overlay(self, p_raw, gdp_now, unemp_now, gdp0, unemp0):
        ov = self.ecl_cfg.get("pd_overlay") or {}
        e_g = float(ov.get("gdp_elasticity", 0.0))
        e_u = float(ov.get("unemp_elasticity", 0.0))
        p = np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(p / (1.0 - p)) + e_g * (gdp_now - gdp0) + e_u * (unemp_now - unemp0)
        return 1.0 / (1.0 + np.exp(-logit))

    def _macro_pair(self, row: pd.Series):
        gdp, unemp = 0.0, 0.0
        if "realgdp_growth" in row.index:
            gdp = float(row["realgdp_growth"])
        else:
            for c in row.index:
                if "gdp" in str(c).lower() and "name" not in str(c).lower():
                    try:
                        gdp = float(row[c])
                        break
                    except (TypeError, ValueError):
                        pass
        if "unemp" in row.index:
            unemp = float(row["unemp"])
        else:
            for c in row.index:
                if "unemp" in str(c).lower():
                    try:
                        unemp = float(row[c])
                        break
                    except (TypeError, ValueError):
                        pass
        return gdp, unemp

    def _stage(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.ecl_cfg
        dpd = df["days_past_due"] if "days_past_due" in df.columns else 0
        orig = pd.to_numeric(df.get("origination_pd"), errors="coerce")
        base = pd.to_numeric(df.get("baseline_pd"), errors="coerce")
        orig_dt = pd.to_datetime(df.get("origination_date"), errors="coerce")
        pre_fred = orig_dt.lt(pd.Timestamp("2000-01-01")).fillna(False)
        # Pre-2000 names have no FRED history; do not SICR them off a junk origination PD.
        anchor = orig.mask(pre_fred, base)
        anchor = anchor.fillna(base).fillna(df["PIT_PD_12M"]).clip(lower=0.05)
        sicr = (df["PIT_PD_12M"] / anchor >= float(cfg.get("sicr_pd_ratio", 2.5))) | (
            df["PIT_PD_12M"] - anchor >= float(cfg.get("sicr_pd_add", 0.08))
        )
        stage = pd.Series(1, index=df.index)
        stage = stage.mask(sicr | (dpd >= int(cfg.get("stage2_dpd", 30))), 2)
        stage3 = (df["PIT_PD_12M"] >= float(cfg.get("stage3_pd", 0.40))) | (dpd >= int(cfg.get("stage3_dpd", 90)))
        return stage.mask(stage3, 3).astype(int)

    def _term_ecl(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.ecl_cfg
        r = float(cfg["discount_rate"])
        h_max = int(cfg["max_horizon_years"])
        floor = float(cfg["hazard_floor"])
        cap = float(cfg["hazard_cap"])
        pd12 = np.clip(df["PIT_PD_12M"].to_numpy(), floor, cap)
        mat = np.clip(df["maturity_years"].to_numpy(), 0.25, h_max)
        lgd = df["Downturn_LGD"].to_numpy()
        ead = df["exposure_at_default"].to_numpy()
        stage = df["ifrs9_stage"].to_numpy()

        n = len(df)
        lifetime = np.zeros(n)
        twelve = np.zeros(n)
        surv = np.ones(n)
        t = 1
        while t <= h_max:
            # Constant annual hazard implied by the 12m PD (no ad-hoc 5% growth).
            haz = np.clip(pd12, floor, cap)
            live = (mat >= (t - 1 + 1e-9)).astype(float)
            frac = np.clip(mat - (t - 1), 0, 1)
            marg = surv * haz * frac * live
            add = marg * lgd * ead / ((1.0 + r) ** t)
            lifetime += add
            if t == 1:
                twelve += add
            surv = surv * (1.0 - haz * frac)
            t += 1

        # Stage 3: cash-shortfall ≈ EAD × downturn LGD (default already crystallized).
        shortfall = ead * lgd
        use = np.where(stage == 1, twelve, lifetime)
        use = np.where(stage == 3, shortfall, use)
        df = df.copy()
        df["ECL_12M"] = twelve
        df["ECL_lifetime"] = lifetime
        df["lifetime_pd"] = 1.0 - surv
        df["ECL_stage3_shortfall"] = shortfall
        df["ECL"] = use
        return df

    def run_stress_test(self) -> dict:
        results = {}
        weighted = 0.0
        base_rows = self.scenarios[self.scenarios["scenario_name"] == "baseline"]
        base_macro = (base_rows.iloc[0] if len(base_rows) else self.scenarios.iloc[0])
        gdp0, unemp0 = self._macro_pair(base_macro)
        base_df = self._map_macro_features(self.portfolio, base_macro)
        baseline_pd = self.pd_model.predict_proba(base_df[self.pd_features])[:, 1]
        # Origination PD = score on the loan's own historical macros (tape), not the scenario.
        orig_X = self.portfolio.copy()
        for c in self.pd_features:
            if c not in orig_X.columns:
                orig_X[c] = 0.0
        origination_pd = self.pd_model.predict_proba(orig_X[self.pd_features])[:, 1]

        for scenario in self.scenarios["scenario_name"].unique():
            macro = self.scenarios[self.scenarios["scenario_name"] == scenario].iloc[0]
            df = self._map_macro_features(self.portfolio, macro)
            raw = self.pd_model.predict_proba(df[self.pd_features])[:, 1]
            gdp_s, unemp_s = self._macro_pair(macro)
            df["PIT_PD_RAW"] = raw
            df["PIT_PD_12M"] = raw
            df["baseline_pd"] = baseline_pd
            df["origination_pd"] = origination_pd
            p_loss = self.lgd_model["classifier"].predict_proba(df[self.lgd_features])[:, 1]
            severity = np.clip(self.lgd_model["regressor"].predict(df[self.lgd_features]), 0.0, 1.0)
            pit_lgd = np.clip(p_loss * severity, 0.0, 1.0)
            df["PIT_LGD"] = pit_lgd
            df["Downturn_LGD"] = np.maximum(pit_lgd, float(self.ecl_cfg["downturn_lgd_floor"]))
            df["ifrs9_stage"] = self._stage(df)
            df = self._term_ecl(df)

            total = float(df["ECL"].sum())
            w = float(self.weights.get(scenario, 0.0))
            weighted += total * w
            ead = float(df["exposure_at_default"].sum())
            s1 = df["ifrs9_stage"] == 1
            s2 = df["ifrs9_stage"] == 2
            s3 = df["ifrs9_stage"] == 3
            results[scenario] = {
                "Total_ECL": total,
                "ECL_12M_sum": float(df["ECL_12M"].sum()),
                "ECL_lifetime_sum": float(df["ECL_lifetime"].sum()),
                "ECL_stage1": float(df.loc[s1, "ECL"].sum()) if s1.any() else 0.0,
                "ECL_stage2": float(df.loc[s2, "ECL"].sum()) if s2.any() else 0.0,
                "ECL_stage3": float(df.loc[s3, "ECL"].sum()) if s3.any() else 0.0,
                "Coverage": total / ead if ead else float("nan"),
                "Average_PD": float(df["PIT_PD_12M"].mean()),
                "Average_PIT_LGD": float(df["PIT_LGD"].mean()),
                "Average_Downturn_LGD": float(df["Downturn_LGD"].mean()),
                "Stage1_share": float(s1.mean()),
                "Stage2_share": float(s2.mean()),
                "Stage3_share": float(s3.mean()),
                "Facility_Data": df,
            }
            print(
                f"  [{scenario}] ECL=${total:,.0f}  cov={total/ead:.2%}  "
                f"PD_raw={df['PIT_PD_RAW'].mean():.2%} PD={df['PIT_PD_12M'].mean():.2%}  "
                f"LGD={df['PIT_LGD'].mean():.2%}  "
                f"S1/S2/S3={s1.mean():.1%}/{s2.mean():.1%}/{s3.mean():.1%}  "
                f"dU={unemp_s-unemp0:+.2f} dG={gdp_s-gdp0:+.3f}"
            )
        results["Probability_Weighted_ECL"] = weighted
        wsum = sum(float(self.weights.get(s, 0)) for s in self.scenarios["scenario_name"].unique())
        results["weight_sum"] = wsum
        return results
