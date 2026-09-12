from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from src.data_prep.sba_credit_pipeline import run_pipeline
from src.ecl_models.ecl_dashboard import ECLVisualizer
from src.ecl_models.ecl_execution_engine import ECLExecutionEngine
from src.ecl_models.train_ecl_model import run_training
from src.evaluation.diagnostics import dump_run_json, print_scenario_book
from src.genAI_validation.sr117_report_generator import SR117ReportGenerator
from src.macro_engine.macro_var_engine import generate_scenarios


def run_all(root: str = ".", genai: bool = False) -> dict:
    root_p = Path(root).resolve()
    os.chdir(root_p)
    tape = run_pipeline(root=str(root_p))
    generate_scenarios(root=str(root_p))
    metrics = run_training(root=str(root_p))

    with open(root_p / "config/config.yaml") as f:
        config = yaml.safe_load(f)

    engine = ECLExecutionEngine(
        portfolio_path=config["paths"]["processed_tape"],
        macro_scenarios_path=config["paths"]["macro_forecasts"],
        root=str(root_p),
    )
    if metrics.get("run_id"):
        engine.run_id = metrics["run_id"]
    print("\n=== STRESS ENGINE ===")
    print("PIT PD = HGB only (no overlay)")
    results = engine.run_stress_test()
    print_scenario_book(results)
    try:
        ECLVisualizer(results).plot_all(show=False)
    except Exception as e:
        print(f"Figures skipped: {e}")

    extras = {}
    key = os.getenv("OPENAI_API_KEY")
    if not genai and key:
        genai = True
        print("OPENAI_API_KEY set — enabling GenAI")
    if not key:
        print("GenAI off: no OPENAI_API_KEY")
    if genai and key:
        try:
            from src.genAI_validation.agent_analyzer import run_agentic_anomaly_analysis
            from src.genAI_validation.rag_auditor import SR117RAGAuditor

            extras["agent"] = run_agentic_anomaly_analysis(results["severe"]["Facility_Data"])
            extras["rag"] = SR117RAGAuditor().audit_model_output("see limitations section")
        except Exception as e:
            extras["agent"] = f"GenAI skipped: {e}"

    source = ""
    if "source" in tape.columns:
        source = str(tape["source"].dropna().astype(str).mode().iloc[0])
    portfolio_metrics = {
        "total_facilities": int(len(tape)),
        "total_ead": float(tape["exposure_at_default"].sum()),
        "source": source or tape.attrs.get("raw_mode", ""),
        "layout": source,
        "default_rate": float(tape["default_flag"].mean()) if "default_flag" in tape.columns else None,
    }
    gen = SR117ReportGenerator(
        results,
        portfolio_metrics,
        holdout_metrics=metrics,
        run_id=engine.run_id,
        extras=extras,
        config=config,
    )
    text = gen.generate_report_text()
    pdf = gen.render_pdf(text)
    slim = {
        "pwecl": results.get("Probability_Weighted_ECL"),
        "metrics": metrics,
        "portfolio": portfolio_metrics,
        "scenarios": {
            k: {kk: vv for kk, vv in block.items() if kk != "Facility_Data"}
            for k, block in results.items()
            if isinstance(block, dict)
        },
    }
    dump_run_json(root_p / "reports" / "last_run.json", slim)
    return {"results": results, "metrics": metrics, "pdf": pdf, "report_text": text}


if __name__ == "__main__":
    out = run_all()
    print("PWECL", out["results"]["Probability_Weighted_ECL"])
    print(out["metrics"])
