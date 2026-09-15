"""Per-round tracking and plotting for RhoEdit sequential editing.

Opt-in from ``run_rhoedit.py --seq_plot_every N`` (default 0 = off, so the base
training path is untouched). When enabled it collects, after every N-th round,
the data behind the paper's sequential-editing figures/tables:

* Figure 2(a)  retention heatmap: edit chunk x round -> reliability
* Figure 2(b)  final-round comparison across methods (via ``summary.json``)
* Table 8      every-N-round Current Rel. / Retained Rel. / Retained Gen. /
               capability-loss drift / Time (h) / Memory (GB) / State (GB)

Reliability inside the loop is a light proxy: teacher-forced token accuracy of
the edit target given the prompt (EasyEdit ``rewrite_acc``). Official WILD /
LLM-judge numbers still come from ``run_edited_benchmarks.py`` on saved models.
Time only counts editing (K-FAC estimation + optimisation), not this evaluation.
"""

import copy
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import torch

from easyeditor.tools import ExperimentTracker

CAP_LOSS_KEY = "General capabilities of the model Loss"


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def _model_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else None


@torch.no_grad()
def teacher_forced_token_acc(
    model,
    tok,
    prompts: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 32,
    max_length: Optional[int] = None,
) -> List[float]:
    """Per-example fraction of target tokens predicted correctly (teacher forcing).

    Mirrors the training tokenisation in ``execute_sft_adam_sequential``:
    ``prompt + target`` is tokenised jointly, prompt length is measured with
    ``add_special_tokens=True``, right padding is assumed.
    """
    device = _model_device(model)
    trunc = {"truncation": True, "max_length": max_length} if max_length else {}
    accs: List[float] = []
    for start in range(0, len(prompts), batch_size):
        b_prompts = list(prompts[start:start + batch_size])
        b_targets = list(targets[start:start + batch_size])
        texts = [p + t for p, t in zip(b_prompts, b_targets)]
        enc = tok(texts, return_tensors="pt", padding=True, **trunc).to(device)
        logits = model(**enc, use_cache=False).logits
        pred = logits[:, :-1].argmax(-1)
        gold = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        for j, prompt in enumerate(b_prompts):
            prompt_len = len(tok(prompt, add_special_tokens=True, **trunc)["input_ids"])
            m = mask[j].clone()
            m[: max(prompt_len - 1, 0)] = False  # pred[t] predicts token t+1
            n = int(m.sum())
            accs.append(float((pred[j][m] == gold[j][m]).float().mean()) if n > 0 else 0.0)
    return accs


def cov_cache_nbytes(cache: Optional[Dict[str, Dict]]) -> int:
    """Bytes of all tensors held in a layer->{A,B,...} K-FAC cache (editor state)."""
    if not cache:
        return 0
    total = 0
    for layer_cache in cache.values():
        for value in layer_cache.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
class SequentialEditTracker:
    """Collects per-round metrics during sequential editing and draws the plots."""

    def __init__(
        self,
        out_dir: str,
        eval_every: int = 1,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        run_name: str = "",
        cache_style: str = "",
    ):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.eval_every = max(int(eval_every), 1)
        self.batch_size = batch_size
        self.max_length = max_length
        self.run_name = run_name
        self.cache_style = cache_style

        self.records: List[Dict] = []
        # chunk index -> {round -> value}
        self.rel_matrix: Dict[int, Dict[int, float]] = {}
        self.gen_matrix: Dict[int, Dict[int, float]] = {}
        self.baseline_cap_loss: Optional[float] = None
        self.num_rounds: Optional[int] = None

        self._round_start: Optional[float] = None
        self._edit_seconds = 0.0

    # ---- lifecycle ------------------------------------------------------- #
    def set_baseline(self, cap_loss: Optional[float], num_rounds: int):
        self.baseline_cap_loss = cap_loss
        self.num_rounds = num_rounds

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def begin_round(self, round_idx: int):
        self._sync()
        if torch.cuda.is_available():
            for dev in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev)
        self._round_start = time.time()

    def _peak_mem_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 1e9

    def end_round(
        self,
        round_idx: int,
        model,
        tok,
        txt_chunks: List[List[str]],
        tgt_chunks: List[List[str]],
        rephrase_chunks: Optional[List[List[Optional[str]]]],
        state_bytes: int,
        cap_loss: Optional[float] = None,
        ft_loss: Optional[float] = None,
    ) -> Dict:
        self._sync()
        if self._round_start is not None:
            self._edit_seconds += time.time() - self._round_start
            self._round_start = None
        round_no = round_idx + 1

        rec: Dict = {
            "round": round_no,
            "edits_seen": int(sum(len(c) for c in txt_chunks)),
            "time_h": self._edit_seconds / 3600.0,
            "peak_mem_gb": self._peak_mem_gb(),
            "state_gb": state_bytes / 1e9,
            "cap_loss": cap_loss,
            "delta_cap_loss": (cap_loss - self.baseline_cap_loss)
            if (cap_loss is not None and self.baseline_cap_loss is not None) else None,
            "ft_loss": ft_loss,
        }

        is_last = self.num_rounds is not None and round_no >= self.num_rounds
        if round_no % self.eval_every == 0 or is_last:
            per_chunk_rel: List[float] = []
            per_chunk_gen: List[Optional[float]] = []
            for c, (txt, tgt) in enumerate(zip(txt_chunks, tgt_chunks)):
                rel = _mean(teacher_forced_token_acc(
                    model, tok, txt, tgt, self.batch_size, self.max_length))
                per_chunk_rel.append(rel)
                self.rel_matrix.setdefault(c, {})[round_no] = rel

                gen = None
                reph = rephrase_chunks[c] if rephrase_chunks else None
                if reph and all(isinstance(r, str) and r for r in reph):
                    gen = _mean(teacher_forced_token_acc(
                        model, tok, reph, tgt, self.batch_size, self.max_length))
                    self.gen_matrix.setdefault(c, {})[round_no] = gen
                per_chunk_gen.append(gen)

            rec.update({
                "current_rel": per_chunk_rel[-1],
                "retained_rel": _mean(per_chunk_rel[:-1]),
                "all_rel": _mean(per_chunk_rel),
                "current_gen": per_chunk_gen[-1],
                "retained_gen": _mean(per_chunk_gen[:-1]),
                "all_gen": _mean(per_chunk_gen),
                "per_chunk_rel": per_chunk_rel,
                "per_chunk_gen": per_chunk_gen,
            })
            print(
                f"[SeqTracker] round {round_no}: current_rel={rec['current_rel']:.3f} "
                f"retained_rel={rec['retained_rel'] if rec['retained_rel'] is None else round(rec['retained_rel'], 3)} "
                f"retained_gen={rec['retained_gen'] if rec['retained_gen'] is None else round(rec['retained_gen'], 3)} "
                f"time={rec['time_h']:.3f}h mem={rec['peak_mem_gb']:.2f}GB state={rec['state_gb']:.2f}GB"
            )

        self.records.append(rec)
        ExperimentTracker.log({
            f"SEQ/{k}": v for k, v in rec.items()
            if isinstance(v, (int, float)) and v is not None
        })
        self.save()
        return rec

    # ---- persistence ----------------------------------------------------- #
    def _rounds_evaluated(self) -> List[int]:
        return sorted({r for row in self.rel_matrix.values() for r in row})

    def _matrix_as_list(self, matrix: Dict[int, Dict[int, float]]) -> List[List[Optional[float]]]:
        rounds = self._rounds_evaluated()
        n_chunks = (max(matrix) + 1) if matrix else 0
        return [[matrix.get(c, {}).get(r) for r in rounds] for c in range(n_chunks)]

    def to_dict(self) -> Dict:
        return {
            "run_name": self.run_name,
            "cache_style": self.cache_style,
            "eval_every": self.eval_every,
            "num_rounds": self.num_rounds,
            "baseline_cap_loss": self.baseline_cap_loss,
            "rounds_evaluated": self._rounds_evaluated(),
            "rel_matrix": self._matrix_as_list(self.rel_matrix),
            "gen_matrix": self._matrix_as_list(self.gen_matrix),
            "records": self.records,
        }

    def summary(self) -> Dict:
        """Final-round numbers consumed by ``plot_final_round_comparison``."""
        evaluated = [r for r in self.records if "current_rel" in r]
        last = evaluated[-1] if evaluated else (self.records[-1] if self.records else {})
        keys = ("round", "current_rel", "retained_rel", "retained_gen", "all_rel",
                "cap_loss", "delta_cap_loss", "time_h", "peak_mem_gb", "state_gb")
        out = {k: last.get(k) for k in keys}
        out["label"] = self.run_name
        out["cache_style"] = self.cache_style
        return out

    def save(self):
        with open(os.path.join(self.out_dir, "sequential_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def finalize(self) -> Dict[str, str]:
        self.save()
        data = self.to_dict()
        paths = {}
        if data["rel_matrix"]:
            paths["heatmap"] = plot_retention_heatmap(
                data["rel_matrix"], data["rounds_evaluated"],
                os.path.join(self.out_dir, "fig2a_retention_heatmap.png"),
                title=f"Retention (reliability) - {self.run_name}",
            )
        if data["gen_matrix"]:
            paths["heatmap_gen"] = plot_retention_heatmap(
                data["gen_matrix"], data["rounds_evaluated"],
                os.path.join(self.out_dir, "fig2a_retention_heatmap_gen.png"),
                title=f"Retention (generalization) - {self.run_name}",
                cbar_label="Generalization (token acc.)",
            )
        if self.records:
            paths["curves"] = plot_round_curves(
                self.records, os.path.join(self.out_dir, "table8_round_curves.png"),
                title=self.run_name,
            )
        print(f"[SeqTracker] wrote {self.out_dir}: {', '.join(os.path.basename(p) for p in paths.values())}")
        return paths


# --------------------------------------------------------------------------- #
# Plotting (matplotlib imported lazily so the base path needs no extra deps)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_retention_heatmap(
    matrix: List[List[Optional[float]]],
    rounds: Sequence[int],
    out_path: str,
    title: Optional[str] = None,
    cbar_label: str = "Reliability (token acc.)",
) -> str:
    """Figure 2(a): rows = edit chunks, columns = rounds, colour = reliability.

    Cells for chunks not yet edited at a given round are left white.
    """
    import numpy as np
    plt = _plt()

    arr = np.array(
        [[np.nan if v is None else v for v in row] for row in matrix], dtype=float
    )
    n_chunks, n_rounds = arr.shape
    fig_w = max(4.5, min(0.28 * n_rounds + 1.8, 14))
    fig_h = max(3.5, min(0.22 * n_chunks + 1.5, 12))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = copy.copy(plt.get_cmap("viridis"))  # works on matplotlib 3.5 and 3.10
    cmap.set_bad("white")
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")

    step = max(1, n_rounds // 15)
    ax.set_xticks(range(0, n_rounds, step))
    ax.set_xticklabels([str(rounds[i]) for i in range(0, n_rounds, step)])
    ystep = max(1, n_chunks // 15)
    ax.set_yticks(range(0, n_chunks, ystep))
    ax.set_yticklabels([str(i + 1) for i in range(0, n_chunks, ystep)])
    ax.set_xlabel("Round")
    ax.set_ylabel("Edit chunk")
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_round_curves(records: List[Dict], out_path: str, title: Optional[str] = None) -> str:
    """Table 8 as curves: reliability/generalization, capability-loss drift,
    cumulative time, peak memory and editor state per round."""
    plt = _plt()

    def series(key):
        xs, ys = [], []
        for r in records:
            v = r.get(key)
            if v is not None:
                xs.append(r["round"])
                ys.append(v)
        return xs, ys

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    for key, label, marker in (
        ("current_rel", "Current Rel.", "o"),
        ("retained_rel", "Retained Rel.", "s"),
        ("retained_gen", "Retained Gen.", "^"),
    ):
        xs, ys = series(key)
        if xs:
            ax.plot(xs, ys, marker=marker, ms=4, label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Round")
    ax.set_ylabel("Token accuracy")
    ax.set_title("Edit retention")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    xs, ys = series("delta_cap_loss")
    if xs:
        ax.plot(xs, ys, marker="o", ms=4, color="tab:red")
        ax.axhline(0.0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel("Delta capability loss (nats)")
    else:
        xs, ys = series("cap_loss")
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, color="tab:red")
        ax.set_ylabel("Capability loss (nats)")
    ax.set_xlabel("Round")
    ax.set_title("Capability drift (wiki LM loss)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    xs, ys = series("time_h")
    if xs:
        ax.plot(xs, ys, marker="o", ms=4, color="tab:green")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative edit time (h)")
    ax.set_title("Time")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    xs, ys = series("peak_mem_gb")
    if xs:
        ax.plot(xs, ys, marker="o", ms=4, label="Peak GPU memory", color="tab:purple")
    xs, ys = series("state_gb")
    if xs:
        ax.plot(xs, ys, marker="s", ms=4, label="Editor state", color="tab:orange")
    ax.set_xlabel("Round")
    ax.set_ylabel("GB")
    ax.set_title("Memory / state")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    from matplotlib.ticker import MaxNLocator
    for a in axes.flat:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_final_round_comparison(
    summaries: Dict[str, Dict],
    out_path: str,
    delta_cap_key: str = "delta_cap",
    title: Optional[str] = None,
) -> str:
    """Figure 2(b): final-round comparison across methods.

    ``summaries`` maps method label -> summary dict (``summary.json`` of a run,
    optionally with an externally computed ``delta_cap`` in benchmark points).
    Left: Current Rel. / Retained Rel. / Retained Gen. as grouped dots.
    Middle: diverging dots of ``delta_cap`` (falls back to ``delta_cap_loss``).
    Right: horizontal bars of retained editor state (GB).
    """
    plt = _plt()
    labels = list(summaries.keys())
    y = list(range(len(labels)))[::-1]  # first method on top

    fig, axes = plt.subplots(1, 3, figsize=(12, max(2.5, 0.6 * len(labels) + 1.2)), sharey=True)

    ax = axes[0]
    for key, name, marker, color in (
        ("current_rel", "Current Rel.", "o", "tab:blue"),
        ("retained_rel", "Retained Rel.", "s", "tab:orange"),
        ("retained_gen", "Retained Gen.", "^", "tab:green"),
    ):
        xs = [summaries[l].get(key) for l in labels]
        pts = [(yy, v) for yy, v in zip(y, xs) if v is not None]
        if pts:
            ax.scatter([v for _, v in pts], [yy for yy, _ in pts],
                       marker=marker, s=60, color=color, label=name, zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score after final round")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Edit quality")

    ax = axes[1]
    use_key = delta_cap_key if any(delta_cap_key in s for s in summaries.values()) else "delta_cap_loss"
    vals = [summaries[l].get(use_key) for l in labels]
    for yy, v in zip(y, vals):
        if v is None:
            continue
        color = "tab:red" if (v < 0 if use_key == delta_cap_key else v > 0) else "tab:green"
        ax.plot([0, v], [yy, yy], color=color, lw=2, alpha=0.6)
        ax.scatter([v], [yy], color=color, s=60, zorder=3)
    ax.axvline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Delta capability (points)" if use_key == delta_cap_key else "Delta capability loss (nats)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Capability change")

    ax = axes[2]
    vals = [summaries[l].get("state_gb") or 0.0 for l in labels]
    ax.barh(y, vals, color="tab:gray")
    for yy, v in zip(y, vals):
        ax.text(v, yy, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Retained editor state (GB)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("State")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
