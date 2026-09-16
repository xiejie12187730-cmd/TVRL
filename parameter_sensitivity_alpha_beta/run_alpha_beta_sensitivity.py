import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import get_args, run_experiment  # noqa: E402


ALPHA_VALUES = [1e-3, 5e-3, 1e-2, 3e-2, 5e-2]
BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
METRIC_NAMES = ["AP", "1-HL", "1-RL", "AUC", "1-OE", "1-Cov"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run TVI-Net alpha-beta sensitivity experiments.")
    parser.add_argument("--dataset-name", type=str, default="corel5k")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHA_VALUES)
    parser.add_argument("--betas", type=float, nargs="+", default=BETA_VALUES)
    parser.add_argument("--mode", choices=["grid", "one-at-a-time"], default="grid")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-times", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def value_label(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_cases(dataset_name, alphas, betas, mode):
    base_args = get_args(dataset_name, argv=[])
    if mode == "grid":
        return list(itertools.product(alphas, betas))

    cases = []
    for alpha in alphas:
        cases.append((alpha, base_args.beta))
    for beta in betas:
        candidate = (base_args.alpha, beta)
        if candidate not in cases:
            cases.append(candidate)
    return cases


def make_experiment_args(cli_args, alpha, beta, log_suffix):
    args = get_args(cli_args.dataset_name, argv=[])
    args.alpha = alpha
    args.beta = beta
    args.vae_loss_weight = alpha
    args.log_suffix = log_suffix

    if cli_args.epochs is not None:
        args.epochs = cli_args.epochs
    if cli_args.run_times is not None:
        args.run_times = cli_args.run_times
    if cli_args.seed is not None:
        args.seed = cli_args.seed

    return args


def flatten_summary(args, alpha, beta, exp_name, combo_dir, result, status="ok", error=""):
    row = {
        "status": status,
        "error": error,
        "experiment": exp_name,
        "dataset": args.dataset_name,
        "alpha": alpha,
        "beta": beta,
        "gamma": args.gamma,
        "lambda_t": args.lambda_t,
        "epochs": args.epochs,
        "run_times": args.run_times,
        "seed": args.seed,
        "log_dir": str(combo_dir / "logs"),
        "record_dir": str(combo_dir / "records"),
    }

    metrics = result.get("metrics", {}) if result else {}
    for name in METRIC_NAMES:
        metric = metrics.get(name, {})
        row[f"{name}_mean"] = metric.get("mean", "")
        row[f"{name}_std"] = metric.get("std", "")

    return row


def json_record(row, result):
    record = dict(row)
    if result:
        record["fold_metrics"] = [
            [float(value) for value in fold_result[: len(METRIC_NAMES)]]
            for fold_result in result.get("final_res", [])
        ]
    return record


def main():
    cli_args = parse_args()
    os.chdir(REPO_ROOT)

    run_root = Path(cli_args.output_dir) if cli_args.output_dir else SCRIPT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    csv_path = run_root / "summary_alpha_beta.csv"
    jsonl_path = run_root / "summary_alpha_beta.jsonl"

    fieldnames = [
        "status",
        "error",
        "experiment",
        "dataset",
        "alpha",
        "beta",
        "gamma",
        "lambda_t",
        "epochs",
        "run_times",
        "seed",
        "AP_mean",
        "AP_std",
        "1-HL_mean",
        "1-HL_std",
        "1-RL_mean",
        "1-RL_std",
        "AUC_mean",
        "AUC_std",
        "1-OE_mean",
        "1-OE_std",
        "1-Cov_mean",
        "1-Cov_std",
        "log_dir",
        "record_dir",
    ]

    cases = build_cases(cli_args.dataset_name, cli_args.alphas, cli_args.betas, cli_args.mode)
    total = len(cases)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, (alpha, beta) in enumerate(cases, start=1):
            exp_name = f"alpha_{value_label(alpha)}_beta_{value_label(beta)}"
            combo_dir = run_root / exp_name
            args = make_experiment_args(cli_args, alpha, beta, exp_name)

            print(f"[{index}/{total}] Running {exp_name}")
            try:
                result = run_experiment(args, log_dir=str(combo_dir / "logs"), record_dir=str(combo_dir / "records"))
                row = flatten_summary(args, alpha, beta, exp_name, combo_dir, result)
            except Exception as exc:
                row = flatten_summary(args, alpha, beta, exp_name, combo_dir, None, status="failed", error=repr(exc))
                writer.writerow(row)
                csv_file.flush()
                jsonl_file.write(json.dumps(json_record(row, None), ensure_ascii=False) + "\n")
                jsonl_file.flush()
                if not cli_args.continue_on_error:
                    raise
            else:
                writer.writerow(row)
                csv_file.flush()
                jsonl_file.write(json.dumps(json_record(row, result), ensure_ascii=False) + "\n")
                jsonl_file.flush()

    print(f"Done. Summary CSV: {csv_path}")
    print(f"Detailed JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
