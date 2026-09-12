
**ECL run summary**
Demo engine. Not a production or validated model.

**Portfolio**
Source: kaggle_sba_national
Facilities: 897,167
EAD: $180,867,103,981.00
PWECL: $6,823,895,968.05
Coverage: 3.77%
MLflow: 21d78d29bcdc4013af7b6a81d69e3e97
Macro lags: ['realgdp_growth_lag_1Q', 'unemp_lag_1Q']

**Method**
PD: 12m HGB + logit overlay vs baseline macros. LGD: hurdle logit x ridge, downturn floor.
ECL: Stage 1 = 12m; Stage 2 = lifetime (constant hazard from PD_12m, cap 5y); Stage 3 = EAD x LGD.
SICR: PIT PD vs origination PD (tape macros). Scenarios: VAR one-step + config shocks.

**Holdout**
HGB   n=269151  DR=18.11%  AUC=0.682  Brier=0.1392  slope=0.988
Logit AUC=0.667  Brier=0.1408

**Scenarios**
Scenario     | ECL              | PIT PD | PIT LGD | S1     | S2     | S3
baseline     | $6,754,141,037.93 | 24.64% |  11.22% |  86.1% |  13.9% |   0.0%
adverse      | $6,861,212,636.30 | 24.85% |  11.72% |  85.8% |  14.2% |   0.0%
severe       | $6,942,308,290.95 | 25.01% |  12.47% |  85.7% |  14.3% |   0.0%

**Limits**
Kaggle proxies only. HGB does not reprice with UNRATE; scenario PD barely moves.
No performing-book 12m default; closed PIF/CHGOFF tape. FRED from 2000 only.
Point scenarios, constant EAD, no CCF. Random holdout. Demo only. GenAI text is not a control.

**Notes**
GenAI skipped: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}
