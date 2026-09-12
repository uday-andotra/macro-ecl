from __future__ import annotations

import os


def run_agentic_anomaly_analysis(severe_facility_df):
    if not os.getenv("OPENAI_API_KEY"):
        return "Agent skipped: OPENAI_API_KEY missing."
    try:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain_openai import ChatOpenAI
    except ImportError:
        return "Agent skipped: langchain-experimental not installed."

    sample = severe_facility_df.sample(n=min(800, len(severe_facility_df)), random_state=42)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_pandas_dataframe_agent(llm, sample, verbose=False, allow_dangerous_code=True)
    return agent.run(
        "Compute share of ECL from ifrs9_stage >= 2 and from days_past_due >= 30. "
        "Two sentences, numbers only from the dataframe."
    )
