"""Re-draw sequential-editing figures from saved tracker artifacts.

Single run (re-draw Fig.2a heatmap + Table 8 curves from sequential_metrics.json):
    python plot_sequential.py --metrics logs/<run-id>/sequential/sequential_metrics.json

Cross-method Fig.2b (final-round comparison) from several runs' summary.json:
    python plot_sequential.py --compare logs/A/sequential/summary.json logs/B/sequential/summary.json \
        --labels RhoEdit-Online CrispEdit-Seq --out fig2b.png \
        --delta_cap RhoEdit-Online=-0.4 CrispEdit-Seq=-2.1

``--delta_cap`` (benchmark points, from run_base_benchmarks.py) is optional; without it the
middle panel falls back to the in-loop wiki-loss drift stored in summary.json.
"""

import argparse
import json
import os

from easyeditor.models.rhoedit.seq_tracker import (
    plot_final_round_comparison,
    plot_retention_heatmap,
    plot_round_curves,
)


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def redraw_single(metrics_path: str, out_dir: str = None):
    data = _load(metrics_path)
    out_dir = out_dir or os.path.dirname(os.path.abspath(metrics_path))
    os.makedirs(out_dir, exist_ok=True)
    name = data.get("run_name", "")
    written = []
    if data.get("rel_matrix"):
        written.append(plot_retention_heatmap(
            data["rel_matrix"], data["rounds_evaluated"],
            os.path.join(out_dir, "fig2a_retention_heatmap.png"),
            title=f"Retention (reliability) - {name}",
        ))
    if data.get("gen_matrix"):
        written.append(plot_retention_heatmap(
            data["gen_matrix"], data["rounds_evaluated"],
            os.path.join(out_dir, "fig2a_retention_heatmap_gen.png"),
            title=f"Retention (generalization) - {name}",
            cbar_label="Generalization (token acc.)",
        ))
    if data.get("records"):
        written.append(plot_round_curves(
            data["records"], os.path.join(out_dir, "table8_round_curves.png"), title=name))
    for p in written:
        print(f"wrote {p}")


def compare(summary_paths, labels, out_path, delta_cap=None, title=None):
    labels = labels or [os.path.basename(os.path.dirname(os.path.dirname(p))) or p for p in summary_paths]
    if len(labels) != len(summary_paths):
        raise SystemExit("--labels must have one entry per --compare path")
    summaries = {}
    for label, path in zip(labels, summary_paths):
        summaries[label] = _load(path)
    for item in delta_cap or []:
        label, _, value = item.partition("=")
        if label not in summaries:
            raise SystemExit(f"--delta_cap label {label!r} not among {list(summaries)}")
        summaries[label]["delta_cap"] = float(value)
    print(f"wrote {plot_final_round_comparison(summaries, out_path, title=title)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=str, help="sequential_metrics.json of one run (re-draw).")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir for --metrics re-draw.")
    parser.add_argument("--compare", nargs="+", help="summary.json files of several runs (Fig.2b).")
    parser.add_argument("--labels", nargs="+", help="Method labels for --compare, same order.")
    parser.add_argument("--delta_cap", nargs="*", help="label=value pairs (benchmark points) for Fig.2b.")
    parser.add_argument("--out", type=str, default="fig2b_final_round_comparison.png")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if not args.metrics and not args.compare:
        parser.error("give --metrics and/or --compare")
    if args.metrics:
        redraw_single(args.metrics, args.out_dir)
    if args.compare:
        compare(args.compare, args.labels, args.out, args.delta_cap, args.title)


if __name__ == "__main__":
    main()
