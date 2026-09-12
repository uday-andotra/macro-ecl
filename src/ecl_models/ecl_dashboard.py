from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


class ECLVisualizer:
    def __init__(self, stress_results, outdir="reports/figures"):
        self.results = stress_results
        self.scenarios = [s for s in stress_results if s not in {"Probability_Weighted_ECL", "weight_sum"}]
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name):
        path = self.outdir / name
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"Wrote {path}")
        return path

    def plot_macro_stress_dashboard(self, show=True):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ecl = [self.results[s]["Total_ECL"] for s in self.scenarios]
        pd_vals = [self.results[s]["Average_PD"] for s in self.scenarios]
        lgd = [self.results[s]["Average_PIT_LGD"] for s in self.scenarios]
        sns.barplot(x=self.scenarios, y=ecl, ax=axes[0], color="#9b2c2c")
        axes[0].set_title("Portfolio ECL ($)")
        sns.barplot(x=self.scenarios, y=pd_vals, ax=axes[1], color="#2b6cb0")
        axes[1].set_title("Average PIT PD (12m)")
        sns.barplot(x=self.scenarios, y=lgd, ax=axes[2], color="#276749")
        axes[2].set_title("Average PIT LGD")
        fig.suptitle("Why three panels: ECL can move via PD, LGD, or both. Read all three.")
        plt.tight_layout()
        self._save(fig, "stress_dashboard.png")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_pd_distribution_shift(self, show=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        for scenario in self.scenarios:
            sns.kdeplot(
                self.results[scenario]["Facility_Data"]["PIT_PD_12M"],
                ax=ax, label=scenario, fill=True, alpha=0.3,
            )
        ax.set_title("12-month PIT PD across scenarios (overlap = weak macro PD sensitivity)")
        ax.set_xlabel("PIT PD")
        ax.legend()
        plt.tight_layout()
        self._save(fig, "pd_shift.png")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_ecl_by_stage(self, show=True):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = self.scenarios
        s1 = [self.results[s].get("ECL_stage1", 0) for s in labels]
        s2 = [self.results[s].get("ECL_stage2", 0) for s in labels]
        s3 = [self.results[s].get("ECL_stage3", 0) for s in labels]
        x = range(len(labels))
        ax.bar(x, s1, label="Stage 1 (12m)")
        ax.bar(x, s2, bottom=s1, label="Stage 2 (lifetime)")
        ax.bar(x, s3, bottom=[a + b for a, b in zip(s1, s2)], label="Stage 3")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title("ECL by IFRS 9-style stage (Stage 3 frozen on Kaggle CHGOFF)")
        ax.legend()
        plt.tight_layout()
        self._save(fig, "ecl_by_stage.png")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_all(self, show=False):
        self.plot_macro_stress_dashboard(show=show)
        self.plot_pd_distribution_shift(show=show)
        self.plot_ecl_by_stage(show=show)
