# TVI-Net Alpha-Beta Parameter Sensitivity

This folder contains the TVI-Net batch runner for the paper-formula parameters:

```text
L = alpha * L_vae + beta * L_rec + gamma * L_con + lambda_t * L_trust
L_trust = L_bce + L_tc + L_unc
```

Default sweep values:

- alpha: 1e-3, 5e-3, 1e-2, 3e-2, 5e-2
- beta: 0.1, 0.3, 0.5, 0.7, 0.9

Run:

```bash
python parameter_sensitivity_alpha_beta/run_alpha_beta_sensitivity.py
```

Outputs are written under `parameter_sensitivity_alpha_beta/runs/<timestamp>/`:

- `summary_alpha_beta.csv`: one row per parameter setting, with mean/std metrics.
- `summary_alpha_beta.jsonl`: detailed metric records, including fold-level values.
- `alpha_*_beta_*/logs/`: training logs for each setting.
- `alpha_*_beta_*/records/`: final record logs for each setting.

Useful quick test:

```bash
python parameter_sensitivity_alpha_beta/run_alpha_beta_sensitivity.py --epochs 1 --run-times 1
```

The training entry point imports `TVINet` from `models/TVINet.py`. Parameter defaults and experiment calculations are unchanged by the naming update.

The one-epoch command above only checks execution; its losses use the initial warm-up weights.
