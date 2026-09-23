#!/usr/bin/env python3
"""ICLR'27 RhoEdit Table 2 / Table 3 experiment driver.

Table 2: capability-set size n on 3,000 ZsRE edits.
Table 3: penalty strength λ on 3,000 ZsRE edits.

Aligned with the current Llama-3-8B sequential setup:
  layers 14-23, lr=5e-5, newton_damping=1e-4, main λ=10, main n=100,
  3-GPU device_map=auto, batch 3,000 ZsRE (K=1).
  n in {0, 10, 50, 100, 500, 1e3, 1e4, 5e4, 1e5}; n=0 is unconstrained FT.
  Subsets are nested prefixes of one FixedRandomSubsetSampler(seed=1) shuffle.
  λ on a log grid; the shared main-text cell is n=100, λ=10.
  Metrics: Edit Avg, Δcap, Time (h), Memory (GB), retained edit-gradient energy.

On the server:

  bash run_table23.sh              # train all cells, then eval
  bash run_table23.sh train
  bash run_table23.sh eval
  python run_table23.py collect    # fill tables from logs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

# Keep in sync with easyeditor.models.rhoedit.utils.KFAC_DATASET_ALIASES
# (duplicated so `collect` does not import torch).
KFAC_DATASET_ALIASES = {
    "zsre": "zsre_mend_3k",
    "zsre3k": "zsre_mend_3k",
    "zsre_3k": "zsre_mend_3k",
    "zsre10k": "zsre_mend_10k",
    "zsre163k": "zsre_mend_163k",
    "counterfact": "counterfact-edit_3k",
    "wiki": "wiki_big_edit_3k",
}


def normalize_kfac_dataset(ds_name: str) -> str:
    if not ds_name:
        return ds_name
    return KFAC_DATASET_ALIASES.get(str(ds_name).lower(), ds_name)


# Paper Table 2 / Table 3 grids. n=0 has no capability curvature.
# Main cell matches run_0.sh: cache_sample_num=100, soft_lambda=10, layers 14-23.
N_GRID = (0, 10, 50, 100, 500, 1_000, 10_000, 50_000, 100_000)
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
MAIN_N = 100
MAIN_LAMBDA = 10.0
EDIT_LAYERS = (14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
DEFAULT_LR = 5e-5
DEFAULT_NEWTON_DAMPING = 1e-4
LAYER_ARG = ",".join(str(i) for i in EDIT_LAYERS)

CAPABILITY_METRICS = {
    "ifeval": ("results", "ifeval", "prompt_level_strict_acc,none"),
    "truthfulqa_mc2": ("results", "truthfulqa_mc2", "acc,none"),
    "mmlu": ("results", "mmlu", "acc,none"),
    "gsm8k_cot": ("results", "gsm8k_cot", "exact_match,flexible-extract"),
    "arc_challenge": ("results", "arc_challenge", "acc_norm,none"),
}

# Table 1 unedited Llama-3-8B-Instruct capability average (percent).
LLAMA3_BASE_CAP_AVG = 65.4


def load_rhoedit_yaml(model: str) -> dict[str, Any]:
    path = Path("hparams") / "RhoEdit" / f"{model}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return config


def default_task_dataset(model_yaml: dict[str, Any], data_type: str) -> str:
    if data_type.lower() in KFAC_DATASET_ALIASES:
        return normalize_kfac_dataset(data_type)
    return normalize_kfac_dataset(model_yaml.get("task_mom2_dataset", data_type))


def run_name(
    model: str,
    data_type: str,
    task_ds: str,
    newton_damping: float,
    soft_lambda: float,
    lr: float,
    n: int,
) -> str:
    # Keep in sync with run_rhoedit.calculate_model_name.
    layer_tag = f"{EDIT_LAYERS[0]}_{EDIT_LAYERS[-1]}"
    name = (
        f"{model}_rhoedit_adam_{data_type}_{task_ds}_"
        f"{newton_damping}_{soft_lambda}_{lr}_n{int(n)}_{layer_tag}"
    )
    return name.replace(".", "_")


def unique_cells(
    table: str, main_lambda: float, main_n: int
) -> list[tuple[int, float, str]]:
    """(n, lambda, which tables this cell fills). Shared main cell is not duplicated."""
    cells: dict[tuple[int, float], set[str]] = {}

    def add(n: int, lam: float, tag: str) -> None:
        cells.setdefault((n, lam), set()).add(tag)

    if table in ("2", "both"):
        for n in N_GRID:
            add(n, main_lambda, "2")
    if table in ("3", "both"):
        for lam in LAMBDA_GRID:
            add(int(main_n), float(lam), "3")

    out = []
    for (n, lam) in sorted(cells, key=lambda kv: (kv[0], kv[1])):
        tags = ",".join(sorted(cells[(n, lam)]))
        out.append((n, lam, tags))
    return out


def train_command(
    python_bin: str,
    model: str,
    data_type: str,
    n: int,
    lam: float,
    lr: float,
    newton_damping: float,
    task_n: int,
    wandb_project: str,
    no_wandb: bool,
) -> str:
    parts = [
        python_bin,
        "run_rhoedit.py",
        "--model",
        model,
        "--data_type",
        data_type,
        "--cache_sample_num",
        str(int(n)),
        "--cache_task_sample_num",
        str(int(task_n)),
        "--batch_size",
        "32",
        "--lr",
        str(float(lr)),
        "--newton_damping",
        str(float(newton_damping)),
        "--soft_lambda",
        str(float(lam)),
        "--layers",
        LAYER_ARG,
        "--disable_old_loss_check",
        "--wandb_project",
        wandb_project,
        "--plat_name",
        "none" if no_wandb else "swanlab",
    ]
    if no_wandb:
        parts.append("--no_wandb")
    return " ".join(parts)


def eval_commands(
    python_bin: str,
    edited_dir: str,
    model: str,
    data_type: str,
    wandb_project: str,
    no_wandb: bool,
) -> list[str]:
    common = [
        "--edited_model_dir",
        edited_dir,
        "--data_type",
        data_type,
        "--alg_name",
        "RhoEdit",
        "--model_name",
        model,
        "--wandb_project",
        wandb_project,
    ]
    extra = ["--no_wandb"] if no_wandb else []
    qa = " ".join(
        [
            python_bin,
            "run_edited_benchmarks.py",
            *common,
            "--context_type",
            "qa_inst",
            "--evaluation_criteria",
            "llm_judge",
            "--eval_num",
            "3000",
            *extra,
        ]
    )
    noc = " ".join(
        [
            python_bin,
            "run_edited_benchmarks.py",
            *common,
            "--context_type",
            "no_context",
            "--evaluation_criteria",
            "llm_judge",
            "--eval_num",
            "3000",
            *extra,
        ]
    )
    cap = " ".join(
        [
            python_bin,
            "run_base_benchmarks.py",
            *common,
            "--tasks",
            "all",
            "--eval_num",
            "200",
            *extra,
        ]
    )
    return [qa, noc, cap]


def _float_get(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    return float(value)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = load_rhoedit_yaml(args.model)
    main_lambda = (
        float(args.main_lambda)
        if args.main_lambda is not None
        else MAIN_LAMBDA
    )
    main_n = int(args.main_n) if args.main_n is not None else MAIN_N
    lr = float(args.lr) if args.lr is not None else DEFAULT_LR
    damping = (
        float(args.newton_damping)
        if args.newton_damping is not None
        else DEFAULT_NEWTON_DAMPING
    )
    task_ds = default_task_dataset(cfg, args.data_type)
    task_n = int(args.cache_task_sample_num)
    jobs = []
    for n, lam, tags in unique_cells(args.table, main_lambda, main_n):
        name = run_name(
            args.model, args.data_type, task_ds, damping, lam, lr, n
        )
        jobs.append(
            {
                "n": n,
                "lambda": lam,
                "tables": tags,
                "main": n == main_n and abs(lam - main_lambda) < 1e-12,
                "run_name": name,
                "train_cmd": train_command(
                    args.python,
                    args.model,
                    args.data_type,
                    n,
                    lam,
                    lr,
                    damping,
                    task_n,
                    args.wandb_project,
                    args.no_wandb,
                ),
                "eval_cmds": eval_commands(
                    args.python,
                    name,
                    args.model,
                    args.data_type,
                    args.wandb_project,
                    args.no_wandb,
                ),
            }
        )
    return jobs


def print_scheduler_lines(jobs: list[dict[str, Any]], stage: str, gpus: int) -> None:
    for job in jobs:
        if stage == "train":
            print(f"{gpus} | {job['train_cmd']}")
        elif stage == "eval":
            for cmd in job["eval_cmds"]:
                print(f"1 | {cmd}")
        else:
            raise ValueError(stage)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _metric(data: dict[str, Any] | None, path: tuple[str, ...]) -> float | None:
    if data is None:
        return None
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(values: Iterable[float | None]) -> float | None:
    items = list(values)
    if not items or any(v is None for v in items):
        return None
    return sum(items) / len(items)


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def collect_one(
    logs: Path, job: dict[str, Any], base_cap_avg: float
) -> dict[str, Any]:
    name = job["run_name"]
    train = _load_json(logs / name / "table23_train.json")
    cap = _load_json(logs / name / "capability.json")
    qa = _load_json(logs / f"{name}_eval_llm_judge_qa_inst" / "mean_metrics.json")
    noc = _load_json(logs / f"{name}_eval_llm_judge_no_context" / "mean_metrics.json")

    cap_vals = [_metric(cap, path) for path in CAPABILITY_METRICS.values()]
    cap_mean = _mean(cap_vals)
    edit_vals = [
        _metric(qa, ("post", "rewrite_acc")),
        _metric(qa, ("post", "rephrase_acc")),
        _metric(noc, ("post", "rewrite_acc")),
        _metric(noc, ("post", "rephrase_acc")),
    ]
    edit_avg = _mean(edit_vals)
    delta_cap = None if cap_mean is None else (100.0 * cap_mean - base_cap_avg)

    return {
        "n": job["n"],
        "lambda": job["lambda"],
        "tables": job["tables"],
        "main": job["main"],
        "run_name": name,
        "edit_avg": None if edit_avg is None else 100.0 * edit_avg,
        "delta_cap": delta_cap,
        "time_h": None if train is None else train.get("time_h"),
        "memory_gb": None if train is None else train.get("memory_gb"),
        "retained_edit_gradient_energy": (
            None if train is None else train.get("retained_edit_gradient_energy")
        ),
        "qa_rel": _metric(qa, ("post", "rewrite_acc")),
        "qa_gen": _metric(qa, ("post", "rephrase_acc")),
        "noc_rel": _metric(noc, ("post", "rewrite_acc")),
        "noc_gen": _metric(noc, ("post", "rephrase_acc")),
        "capability_mean": cap_mean,
    }


def render_table2(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Table 2: Sensitivity to capability-set size n (3,000 ZsRE edits)",
        "",
        "| n | Avg. | Δcap | Time (h) | Memory (GB) |",
        "|---|------|------|----------|-------------|",
    ]
    by_n = {row["n"]: row for row in rows if "2" in row["tables"]}
    for n in N_GRID:
        row = by_n.get(n)
        mark = " **" if row and row["main"] else ""
        if row is None:
            lines.append(f"| {n}{mark} | - | - | - | - |")
            continue
        lines.append(
            f"| {n}{mark} | {_fmt(row['edit_avg'], 1)} | {_fmt(row['delta_cap'], 1)} "
            f"| {_fmt(row['time_h'], 2)} | {_fmt(row['memory_gb'], 1)} |"
        )
    lines.append("")
    lines.append("Shaded / `**` row is the main-text configuration (n=100, layers 14-23).")
    return "\n".join(lines)


def render_table3(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Table 3: Sensitivity to capability-penalty strength λ (3,000 ZsRE edits)",
        "",
        "| λ | Avg. | Δcap | Retained edit-gradient energy |",
        "|---|------|------|------------------------------|",
    ]
    by_l = {row["lambda"]: row for row in rows if "3" in row["tables"]}
    for lam in LAMBDA_GRID:
        row = by_l.get(float(lam))
        if row is None:
            row = min(by_l.values(), key=lambda r: abs(r["lambda"] - lam), default=None)
            if row is None or abs(row["lambda"] - lam) > 1e-12:
                lines.append(f"| {lam:g} | - | - | - |")
                continue
        mark = " **" if row["main"] else ""
        lines.append(
            f"| {lam:g}{mark} | {_fmt(row['edit_avg'], 1)} | {_fmt(row['delta_cap'], 1)} "
            f"| {_fmt(row['retained_edit_gradient_energy'], 3)} |"
        )
    lines.append("")
    lines.append("Shaded / `**` row is the main-text λ=10 (same as run_0.sh).")
    return "\n".join(lines)


def collect(args: argparse.Namespace) -> int:
    jobs = build_jobs(args)
    logs = Path(args.logs)
    rows = [collect_one(logs, job, args.base_cap_avg) for job in jobs]
    payload = {
        "table2": [row for row in rows if "2" in row["tables"]],
        "table3": [row for row in rows if "3" in row["tables"]],
        "base_cap_avg": args.base_cap_avg,
    }
    logs.mkdir(parents=True, exist_ok=True)
    out_json = logs / "table23_results.json"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    text = "\n\n".join([render_table2(rows), render_table3(rows), f"Wrote {out_json}"])
    out_md = logs / "table23_results.md"
    out_md.write_text(text + "\n", encoding="utf-8")
    print(text)
    missing = [
        row["run_name"]
        for row in rows
        if row["time_h"] is None or row["edit_avg"] is None or row["delta_cap"] is None
    ]
    if missing:
        print("\nIncomplete runs (train json and/or eval artifacts missing):")
        for name in missing:
            print(f"  {name}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["print-train", "print-eval", "print-names", "collect"],
        help="print-train/print-eval emit scheduler lines; collect fills the tables.",
    )
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--data_type", default="zsre", choices=["zsre", "zsre10k"])
    parser.add_argument("--table", default="both", choices=["2", "3", "both"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-gpus", type=int, default=3)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--newton_damping", type=float, default=DEFAULT_NEWTON_DAMPING)
    parser.add_argument("--main-lambda", type=float, default=MAIN_LAMBDA)
    parser.add_argument("--main-n", type=int, default=MAIN_N)
    parser.add_argument("--cache_task_sample_num", type=int, default=3000)
    parser.add_argument("--wandb_project", default="RhoEdit_table23")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--logs", default="logs")
    parser.add_argument(
        "--base-cap-avg",
        type=float,
        default=LLAMA3_BASE_CAP_AVG,
        help="Unedited capability average in percent (Table 1 Llama = 65.4).",
    )
    args = parser.parse_args()
    if args.enable_wandb:
        args.no_wandb = False
    return args


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)
    if args.action == "print-train":
        print_scheduler_lines(jobs, "train", args.train_gpus)
        return 0
    if args.action == "print-eval":
        print_scheduler_lines(jobs, "eval", 1)
        return 0
    if args.action == "print-names":
        for job in jobs:
            flag = " MAIN" if job["main"] else ""
            print(
                f"table={job['tables']} n={job['n']} lambda={job['lambda']:g} "
                f"{job['run_name']}{flag}"
            )
        return 0
    if args.action == "collect":
        return collect(args)
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
