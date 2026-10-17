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

顺序编辑（默认使用 `sequential` 缓存策略，即 Algorithm 1）：

```bash
python run_rhoedit.py --model qwen2.5-7b --data_type zsre \
  --sequential_edit --num_edits 100
```

顺序编辑的实验图表（默认关闭）：加 `--seq_plot_every N` 后每 N 轮在当前及所有历史 chunk 上评估（teacher-forced token 准确率、paraphrase 泛化、wiki loss 漂移、累计编辑时间、峰值显存、编辑器状态大小），结束时输出到 `./logs/<run-id>/sequential/`：`fig2a_retention_heatmap.png`、`table8_round_curves.png`、`sequential_metrics.json`、`summary.json`。跨方法的 Fig.2b 用 `python plot_sequential.py --compare <run-A>/summary.json <run-B>/summary.json --labels A B`。

常用覆盖：`--lr`、`--newton_damping`、`--soft_lambda`（不传则用 YAML）。模型保存到 `${HF_CACHE_DIR}<run-id>/`。

### 顺序编辑：历史事实回放

历史回放默认关闭。启用后，每一轮会从已经完成的历史批次中抽取 prompt/target，并与当前批次共同计算编辑损失；RhoEdit 的 K-FAC 曲率过滤仍照常应用。`--history_replay_ratio` 是每个当前 optimizer batch 对应的历史样本比例，`--history_replay_max` 是每个 batch 的上限（`0` 表示不设上限），`--history_replay_weight` 是历史损失相对当前损失的权重。第一轮没有历史样本，因此不会回放。

例如每个当前 batch 使用约 25% 的历史样本：

```bash
python run_rhoedit.py --model llama3-8b --data_type zsre \
  --sequential_edit --num_edits 100 --edit_cache_style sequential \
  --history_replay_ratio 0.25 --history_replay_max 16 \
  --history_replay_weight 1.0 --plat_name none --no_wandb
```

历史回放保存的是已完成轮次的全部 prompt/target（不检查是否编辑成功），不保存旧模型副本。若后续编辑明确修改同一事实，需要在数据层标记旧目标失效，否则旧目标会与新目标竞争。启用后日志会记录历史样本数、历史损失和回放配置。

CPU 验证（需要 PyTorch 和 python-dotenv，无需下载模型）：

```bash
python -m unittest discover -s tests -p test_history_replay.py -v
```

### 顺序编辑：CrispEdit 风格的 mix 缓存

已有的 `--edit_cache_style mix` 现在通过独立的 `refresh_mix_cache` 路径实现并有跨方法流程测试；无需增加新的模式名称。默认仍为 `sequential`。

```bash
python run_rhoedit.py --model llama3-8b --data_type zsre \
  --sequential_edit --num_edits 100 --batch_size 32 \
  --edit_cache_style mix --cache_sample_num 100 --edit_sample_num 1000 \
  --history_replay_ratio 0 --plat_name none --no_wandb
```

每轮训练前，用当前完整编辑块估计编辑侧曲率：`num_edits=100` 时使用当前 100 条，最后不足 100 条时使用实际条数，不混入历史数据或 Wikipedia。第一轮保护侧使用初始 Wikipedia 因子。每轮结束后，将本轮加入历史池，按 CrispEdit 的同一个底层统计函数，从所有已完成编辑中抽样并混入 Wikipedia，使用当前模型重新估计保护侧 A/B，直接替换上一轮保护缓存，不对旧矩阵做累计平均。

下一轮开始时，将新保护因子与该轮新估计的编辑因子配对，只安装一次优化器缓存，并按新相对曲率过滤已有一阶动量；优化器实例继续复用。mix 不加载初始全任务编辑曲率，`cache_task_sample_num` 和 `task_mom2_dataset` 不决定该模式的编辑侧。两侧曲率在一轮训练期间固定；`--recalculate_cache` 在 mix 下不触发轮内刷新，传入时会打印说明。`retention_weight` 只作用于 `sequential`，不参与 mix 的合并。

`edit_sample_num` 既用于历史池抽样，也用于混入 Wikipedia 后的再次抽样，与当前 CrispEdit 一致；不能将其理解为保证处理指定数量的历史样本再额外处理全部 Wikipedia 样本。`cache_sample_num` 控制初始能力统计及每轮混合的 Wikipedia 候选数。mix 只用历史数据估计曲率，不会自动启用历史损失回放。保护侧流程一致不代表两种方法的完整更新相同，编辑层、梯度过滤和训练截断等仍需单独对齐。

CPU 流程测试（无需下载模型）：

```bash
python -m unittest discover -s tests -p test_mix_cache.py -v
```

### 历史回放消融

显式 K-FAC 参数正则已删除，旧命令中的 `--reg_lambda` 需要移除。`soft_lambda` 仍控制 RhoEdit 原有的相对曲率过滤，两组应保持一致。

在相同启动命令上分别使用以下参数：

| 对照组 | 参数 |
| --- | --- |
| RhoEdit | `--sequential_fixed_steps --history_replay_ratio 0` |
| RhoEdit + 回放 | `--sequential_fixed_steps --history_replay_ratio 0.25 --history_replay_max 16 --history_replay_weight 1` |

两组都要开启 `--sequential_edit`。`--sequential_fixed_steps` 在每轮禁用低损失跳步和提前停止；每轮实际更新次数为 `num_steps * ceil(当前轮条数 / batch_size)`。每轮记录 `Sequential/optimizer_steps`，输出名称增加 `_fixedsteps`。未设置该开关时沿用旧策略：没有回放的轮次仍可提前停止，有回放的轮次执行完整步数。回放权重为 0 时不执行回放。

当前回放目标为 `L = (L_current + s * L_history) / (1 + s)`，其中 `s = history_replay_weight * 历史批大小 / 当前批大小`，不含参数二次惩罚；每项 loss 是各自模型返回的 token 平均损失。两组须匹配模型初始权重、编辑顺序、学习率、曲率配置和评测方式。固定步数消除了更新次数差异，但回放仍增加样本量和计算量；若要进一步区分历史内容与额外数据处理的作用，需要补充等计算预算对照。历史数据可能包含冲突目标，当前采样不自动处理版本替换。

### K-FAC 统计分配到两张 GPU

现支持三卡：添加 `--kfac_stats_devices auto_three` 自动选择空闲显存最多的三张可见卡，或用 `--kfac_stats_devices 0,1,2` 固定逻辑 GPU。十层按 4/3/3 分组；float32 下每组常驻统计矩阵约 3.31/2.48/2.48 GiB。该值不包括模型、激活、梯度、Adam 状态和分解临时空间。默认仍为 `auto_two`；自动选择只按空闲显存排序，不保证所选卡显存足够。

例如三张充足空闲的物理卡为 0、6、7 时，可在原训练命令前加 `CUDA_VISIBLE_DEVICES=0,6,7`，并在参数中加 `--kfac_stats_devices 0,1,2`。卡号是示例，启动前须确认可用性。模型仍由 `device_map="auto"` 分配，该参数只改变统计矩阵的位置，不保证模型权重或训练负载平均分到三卡。

RhoEdit 的 `kfac_stats_devices` 默认是 `auto_two`：在每次需要重新估计曲率时，选择当前进程可见且空闲显存最多的两张 GPU，将按顺序排列的编辑层分成两组。Llama 的 14–23 层分为 14–18、19–23，每组五层；每层的 A/B 因子保持独立。初始化能力/任务统计、每轮编辑前和编辑后统计均使用该策略，已有 CPU 缓存可直接复用。

可通过 `--kfac_stats_devices 0,1` 固定两张**逻辑 GPU**。例如 `CUDA_VISIBLE_DEVICES=1,6,7` 下，`0,1` 表示物理卡 1 和 6。`--kfac_stats_devices model` 恢复旧的单设备统计；CUDA 环境少于两张可见卡而选择两卡模式时会明确报错。CPU 模型在 `auto_two` 下继续使用 CPU。

float32 下，十层 down-projection 因子的常驻矩阵从单卡约 8.28 GiB 分散为每卡约 4.14 GiB。该数值不包含模型、梯度、Adam 状态、激活和临时空间。采集到的有效激活/梯度按层搬运到统计卡，输出统计量转回 CPU；不改变曲率定义、采样或模型权重的 `device_map`。这是统计设备设置，与 `factor_device` 控制的特征分解设备不同，也不会阻止其他进程占用 GPU。

包含两卡真实收集流程的测试（不足两张 CUDA 卡时跳过 GPU 测试）：

```bash
python -m unittest discover -s tests -p test_kfac_devices.py -v
```

K-FAC 分解默认 `--factor_device auto`：优先使用训练设备，CUDA 显存不足时对该因子回退 CPU，并打印提示。大矩阵不再按尺寸强制切到 CPU。`--factor_device cuda` 强制使用训练 GPU，显存不足直接报错；`--factor_device cpu` 强制在 CPU 分解。分解仍使用 float64，缓存基底保持在 CPU。

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

编辑质量评测仍使用一个入口脚本。主进程启动独立的 Python 子进程加载 vLLM、生成答案并保存到 `results_pending_judge.json`；等待子进程退出，并在 Linux 上清理、确认同一进程组的工作进程退出后，才在主进程执行 `llm_judge` 和汇总。主进程在导入模型相关库前隐藏 GPU，生成子进程继承启动命令原有的 `CUDA_VISIBLE_DEVICES`。这使生成进程的 CUDA 上下文随进程退出释放；不代表整张 GPU 的总显存占用必然为零。

原有参数和裁判评分、打印逻辑保持兼容。使用 `--evaluation_criteria llm_judge` 即可启用远程裁判；`exact_match` 也通过子进程生成，再由主进程汇总。无需拆成两个程序或修改调用脚本。

新增 `generation_state.json` 记录本次生成的完成状态、参数、条数和文件校验值。生成失败、退出异常或结果校验失败时不会继续调用裁判。Linux 上同一结果目录禁止并发评测。进程组清理面向本地 vLLM worker，不涵盖远程 Ray worker 或自行脱离进程组的进程；Windows 仅清理直接子进程。

如果生成已完成、裁判中途失败，可以在**原命令末尾追加 `--judge_only`**，直接读取已保存的答案而不启动生成子进程。模型目录、数据集、上下文、评测方式、样本数和生成长度等生成参数必须与原命令一致。该选项会重新评判全部已生成答案，不恢复尚未保存的裁判打分；旧版本只有答案文件而没有完成记录的输出不能直接使用此选项。

汇总：`python skills/analyze-results/collect_runs.py`

## 对比方法

CrispEdit / FT / LoRA 走 `run_crispedit.py`，其余基线走 `edit.py --editing_method ...`，配置在 `hparams/<Method>/`。
