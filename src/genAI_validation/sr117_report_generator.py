from __future__ import annotations

import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


class SR117ReportGenerator:
    def __init__(
        self,
        stress_results,
        portfolio_metrics,
        holdout_metrics=None,
        run_id=None,
        extras=None,
        config=None,
    ):
        self.stress_results = stress_results
        self.portfolio_metrics = portfolio_metrics or {}
        self.holdout_metrics = holdout_metrics or {}
        self.run_id = run_id or "Not Logged"
        self.extras = extras or {}
        self.config = config or {}
        self.styles = getSampleStyleSheet()

    def _table(self) -> str:
        lines = ["Scenario     | ECL              | PIT PD | PIT LGD | S1     | S2     | S3"]
        for scenario, metrics in self.stress_results.items():
            if scenario in {"Probability_Weighted_ECL", "weight_sum"}:
                continue
            lines.append(
                f"{scenario:<12} | ${metrics['Total_ECL']:>14,.2f} | "
                f"{metrics['Average_PD']*100:>5.2f}% | "
                f"{metrics['Average_PIT_LGD']*100:>6.2f}% | "
                f"{metrics.get('Stage1_share', 0):>6.1%} | "
                f"{metrics.get('Stage2_share', 0):>6.1%} | "
                f"{metrics.get('Stage3_share', 0):>6.1%}"
            )
        return "\n".join(lines)

    def generate_report_text(self) -> str:
        w = float(self.stress_results.get("Probability_Weighted_ECL", 0) or 0)
        ead = float(self.portfolio_metrics.get("total_ead", 0) or 0)
        n = int(self.portfolio_metrics.get("total_facilities", 0) or 0)
        m = self.holdout_metrics.get("model", {})
        c = self.holdout_metrics.get("challenger_logit", {})
        src = self.portfolio_metrics.get("source") or "unknown"
        lags = self.config.get("model_features", {}).get("macro_lags", [])
        extras = []
        if self.extras.get("narrative"):
            extras.append(self.extras["narrative"])
        if self.extras.get("agent"):
            extras.append(self.extras["agent"])
        extra_block = "\n".join(extras) if extras else "None."

        return f"""
**ECL run summary**
Demo engine. Not a production or validated model.

**Portfolio**
Source: {src}
Facilities: {n:,}
EAD: ${ead:,.2f}
PWECL: ${w:,.2f}
Coverage: {(w / ead) if ead else float('nan'):.2%}
MLflow: {self.run_id}
Macro lags: {lags}

**Method**
PD: 12m HGB + logit overlay vs baseline macros. LGD: hurdle logit x ridge, downturn floor.
ECL: Stage 1 = 12m; Stage 2 = lifetime (constant hazard from PD_12m, cap 5y); Stage 3 = EAD x LGD.
SICR: PIT PD vs origination PD (tape macros). Scenarios: VAR one-step + config shocks.

**Holdout**
HGB   n={m.get('n')}  DR={m.get('default_rate', float('nan')):.2%}  AUC={m.get('auc', float('nan')):.3f}  Brier={m.get('brier', float('nan')):.4f}  slope={m.get('calibration_slope', float('nan')):.3f}
Logit AUC={c.get('auc', float('nan')):.3f}  Brier={c.get('brier', float('nan')):.4f}

**Scenarios**
{self._table()}

**Limits**
Kaggle proxies only. HGB does not reprice with UNRATE; scenario PD barely moves.
No performing-book 12m default; closed PIF/CHGOFF tape. FRED from 2000 only.
Point scenarios, constant EAD, no CCF. Random holdout. Demo only. GenAI text is not a control.

**Notes**
{extra_block}
"""

    def render_pdf(self, report_text, filename="ecl_macro_SR117_Final.pdf"):
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", filename)
        md_path = os.path.join("reports", filename.replace(".pdf", ".md"))
        with open(md_path, "w") as f:
            f.write(report_text)

        doc = SimpleDocTemplate(path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=50, bottomMargin=50)
        body = ParagraphStyle("Body", parent=self.styles["Normal"], fontSize=10, leading=13, spaceAfter=6)
        heading = ParagraphStyle("Heading", parent=self.styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
        mono = ParagraphStyle("Code", parent=self.styles["Normal"], fontName="Courier", fontSize=8, leading=11)
        story = []
        for raw in report_text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("**"):
                story.append(Paragraph(line.replace("**", ""), heading))
                story.append(Spacer(1, 2))
            elif "|" in line:
                story.append(Paragraph(line.replace(" ", "&nbsp;"), mono))
            else:
                story.append(Paragraph(line.replace("&", "&amp;"), body))
        doc.build(story)
        print(f"Wrote {path} and {md_path}")
        return path
