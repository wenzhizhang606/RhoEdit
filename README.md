# RhoEdit

知识编辑：改事实，同时尽量保住原有能力。主方法是 **RhoEdit**（K-FAC 曲率 + 软约束预条件 + Adam）。仓库里还留着 CrispEdit、FT、LoRA、MEND、MEMIT 等对比方法。

## 环境

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` 里需要本机路径：`HF_CACHE_DIR`（必须以 `/` 结尾）、`HF_DATASETS_DIR`、`STATS_DIR`、`EDIT_DATA_DIR`。模型需已下载到本地。LLM-as-a-Judge 才需要 API 相关变量。

## 跑 RhoEdit

`--model` 对应 `hparams/RhoEdit/<model>.yaml`。`--data_type` 同时决定编辑数据和 task K-FAC（`zsre` → `zsre_mend_3k` 等）。`--task_mom2_dataset yaml` 可强制用 YAML 里的任务统计。

```bash
CUDA_VISIBLE_DEVICES=0 python run_rhoedit.py \
  --model llama3-8b \
  --data_type counterfact \
  --cache_sample_num 10000 \
  --cache_task_sample_num 3000 \
  --batch_size 32 \
  --plat_name none \
  --no_wandb
```

顺序编辑：

```bash
python run_rhoedit.py --model qwen2.5-7b --data_type zsre \
  --sequential_edit --num_edits 100 --edit_cache_style mix
```

顺序编辑的实验图表（默认关闭）：加 `--seq_plot_every N` 后每 N 轮在当前及所有历史 chunk 上评估（teacher-forced token 准确率、paraphrase 泛化、wiki loss 漂移、累计编辑时间、峰值显存、编辑器状态大小），结束时输出到 `./logs/<run-id>/sequential/`：`fig2a_retention_heatmap.png`、`table8_round_curves.png`、`sequential_metrics.json`、`summary.json`。跨方法的 Fig.2b 用 `python plot_sequential.py --compare <run-A>/summary.json <run-B>/summary.json --labels A B`。

常用覆盖：`--lr`、`--newton_damping`、`--soft_lambda`（不传则用 YAML）。模型保存到 `${HF_CACHE_DIR}<run-id>/`。

## 评测

原有能力：

```bash
python run_base_benchmarks.py \
  --edited_model_dir <run-id> --model_name llama3-8b \
  --alg_name RhoEdit --data_type counterfact --tasks all --eval_num 200 --no_wandb
```

编辑质量：

```bash
python run_edited_benchmarks.py \
  --edited_model_dir <run-id> --model_name llama3-8b \
  --alg_name RhoEdit --data_type counterfact \
  --context_type qa_inst --evaluation_criteria exact_match --eval_num 3000 --no_wandb
```

汇总：`python skills/analyze-results/collect_runs.py`

## 对比方法

CrispEdit / FT / LoRA 走 `run_crispedit.py`，其余基线走 `edit.py --editing_method ...`，配置在 `hparams/<Method>/`。
