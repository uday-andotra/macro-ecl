from __future__ import annotations

import os

import yaml


def draft_synthetic_scenario(natural_language_shock: str, config_path: str = "config/config.yaml"):
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise ImportError("LangChain + pydantic required for drafting") from e

    class Shock(BaseModel):
        scenario_name: str = Field(description="snake_case scenario name")
        gdp_shock: float = Field(description="additive shock to real GDP growth, e.g. -0.05")
        unemp_shock: float = Field(description="additive shock to unemployment rate, e.g. 0.06")
        weight: float = Field(default=0.0, description="optional probability weight; 0 means report-only")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Shock)
    parsed = llm.invoke(
        "Translate this crisis into additive shocks vs a baseline forecast. "
        f"Scenario: {natural_language_shock}"
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("stress_scenarios", {})[parsed.scenario_name] = {
        "gdp_shock": float(parsed.gdp_shock),
        "unemp_shock": float(parsed.unemp_shock),
    }
    if parsed.weight and parsed.weight > 0:
        config.setdefault("scenario_weights", {})[parsed.scenario_name] = float(parsed.weight)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return parsed.model_dump()
