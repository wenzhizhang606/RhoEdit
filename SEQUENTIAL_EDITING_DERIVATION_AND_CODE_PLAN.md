# 顺序 RhoEdit：历史曲率正则化推导与代码修改指南

日期：2026-10-10。

**本文档仅给出数学推导、拟议接口和代码修改步骤；本次没有修改任何源代码或训练配置。文中的新增参数、函数、伪代码和命令均未实现。**

论文依据：`C:/Users/29308/Desktop/11378_RhoEdit_Balancing_Model_.pdf`（18 页）。原稿 Eq. (3)–(6)、Eq. (11)–(17)、Appendix B.1、C.2、D.1 是本扩展的出发点。当前 PDF 的 Algorithm 1 是单个编辑集合的固定曲率算法，不是代码注释所称的顺序 Algorithm 1；原稿 Eq. (13) 也不是历史因子更新式。

## 1. 实施结论与范围

推荐保留现有顺序流程，增加两个明确分离的模式：

| 模式 | 优化器输入 | 优化器 | 用途 |
|---|---|---|---|
| `relative` | 由相对曲率算子过滤的编辑梯度 | `ProjectedAdam` | 顺序 RhoEdit 主方法 |
| `explicit` | 编辑损失与历史二次正则项的总梯度 | 普通 `torch.optim.Adam` | 显式正则化对照 |
| `none` | 原始编辑梯度 | 普通 `torch.optim.Adam` | 无保护基线，可后续补充 |

不要默认把显式正则项和相对曲率过滤同时开启。二者源自同一局部二次模型，但在 Adam 下不是等价算法；叠加后成为第三种方法，需要独立定义和调参。

状态策略与更新算法分开：`retention_update=fixed` 只使用初始能力状态；`retention_update=post` 将每轮编辑后的因子并入状态。这样能够独立研究动态历史保护的贡献。

优先做 `relative/post`、`relative/fixed`、`explicit/post`，再补仅保留侧几何和实际更新范数匹配的基线。

## 2. 当前代码已经有什么

以下位置以本次读取版本为准，实际修改时按函数名定位：

| 文件/函数 | 已有内容 | 需要补充 |
|---|---|---|
| `run_rhoedit.py::get_arguments/get_hparams` | 顺序开关、chunk 大小、缓存策略、token/sample 权重 | 正则模式、状态更新策略、种子、评估开关等统一参数 |
| `utils.py::execute_sft_adam_sequential` | 按 chunk 训练，轮前估计编辑曲率，轮后重估并合并 | 显式正则化分支、固定轮次锚点、清楚的优化器状态策略 |
| `utils.py::pair_retention_with_edit` | 将保留因子与当前编辑因子配对 | 主方法复用；显式分支不需要强行配对 |
| `utils.py::update_retention_eq13` | 对历史因子进行加权合并 | 规范命名、计数、缓存失效和固定状态对照 |
| `projected_adam.py::step` | 过滤梯度后调用 Adam | 主路径可复用；不能拿来运行纯显式正则化对照 |
| `projected_adam.py::reset_cache` | 更换缓存并再次过滤一阶动量 | 显式记录该行为；它不是单纯坐标变换，也没有同步变换二阶动量 |
| `seq_tracker.py::SequentialEditTracker` | 历史批次 token 准确率、曲线和状态统计 | 入口实际接线、正式生成指标、正则量和分项成本 |

当前 `run_rhoedit.py` 没有 `--seq_plot_every` 参数，也没有构造并传入 tracker；README/模块注释提到的入口能力不能当作已经接通。

## 3. 符号和每轮目标

设编辑任务依次到达为

\[
\mathcal D_1,\mathcal D_2,\ldots,\mathcal D_K.
\]

模型最初参数为 \(\theta_0\)，第 \(k\) 轮起点为 \(\theta_{k-1}\)，编辑完成后为 \(\theta_k\)。本轮编辑损失为 \(\mathcal L_k(\theta)\)。所有曲率和参数只针对允许编辑的权重块定义，其他参数保持固定。

记

\[
\Delta_k(\theta)=\theta-\theta_{k-1},
\qquad C_{\mathrm{ret},k-1}\succeq0.
\tag{S1}
\]

\(C_{\mathrm{ret},k-1}\) 是能力参考数据和历史编辑构成的保留几何代理。在整个第 \(k\) 轮内部，它与锚点 \(\theta_{k-1}\) 都固定。

定义本轮二次正则项和正则化目标：

\[
R_k(\theta)=\frac12\Delta_k(\theta)^\top
C_{\mathrm{ret},k-1}\Delta_k(\theta),
\tag{S2}
\]

\[
J_k(\theta)=\mathcal L_k(\theta)+\lambda R_k(\theta),
\qquad \lambda\ge0.
\tag{S3}
\]

它惩罚的是本轮从起点累计产生的敏感方向位移，不能每个 mini-batch 更新一次锚点，也不能在 optimizer.step 之后把位移归零。

\(\lambda\) 是保护强度；`newton_damping` 用于数值稳定；Adam `weight_decay` 通常约束当前参数而非相对轮次锚点的位移。三者必须分开解释。

### 3.1 与预测分布保持的关系

理想情况下，在由能力上下文和已到达历史上下文构成的分布 \(q_{k-1}\) 上，以当前轮起点作为教师，定义

\[
\mathcal R_k^{\mathrm{KL}}(\theta)
=\mathbb E_{x\sim q_{k-1}}
D_{\mathrm{KL}}\left(p_{\theta_{k-1}}(\cdot|x)
\Vert p_\theta(\cdot|x)\right).
\tag{S4}
\]

在 \(\theta=\theta_{k-1}\) 处，该量及一阶梯度为零，二阶展开为

\[
\mathcal R_k^{\mathrm{KL}}(\theta_{k-1}+\Delta)
=\frac12\Delta^\top F(q_{k-1},\theta_{k-1})\Delta
+o(\|\Delta\|^2).
\tag{S5}
\]

因此，S2 可解释为对局部输出变化的二次代理。实际的历史状态可能由多个旧参数时刻的 K-FAC 因子压缩而来，故

\[
C_{\mathrm{ret},k-1}\approx F(q_{k-1},\theta_{k-1})
\]

是额外近似，不是由递推自动成立的恒等式。

特别注意：若教师固定为原模型 \(\theta_0\)，或历史编辑完成时的 \(\theta_j\)，其 KL 在 \(\theta_{k-1}\) 处的梯度一般不为零，不能省略一阶项并直接套用 S5。

## 4. 从二次正则化得到相对曲率调整

在本轮起点定义 \(g_k=\nabla\mathcal L_k(\theta_{k-1})\)。用正定代理 \(C_{\mathrm{edit},k}\succ0\) 近似编辑曲率；这可以是带阻尼的 GGN/K-FAC 代理，而不必等同于原始非凸 Hessian。

本轮局部问题为

\[
\min_\Delta\;g_k^\top\Delta
+\frac12\Delta^\top C_{\mathrm{edit},k}\Delta
+\frac\lambda2\Delta^\top C_{\mathrm{ret},k-1}\Delta.
\tag{S6}
\]

令梯度为零，得到唯一局部二次解：

\[
\Delta_k^\star
=-(C_{\mathrm{edit},k}+\lambda C_{\mathrm{ret},k-1})^{-1}g_k.
\tag{S7}
\]

分解为

\[
\Delta_k^\star=-C_{\mathrm{edit},k}^{-1}\mathcal A_k g_k,
\qquad
\mathcal A_k=(I+\lambda C_{\mathrm{ret},k-1}C_{\mathrm{edit},k}^{-1})^{-1}.
\tag{S8}
\]

这与原论文 Eq. (4)–(5) 结构一致，只是保护侧改为动态保留状态。

### 4.1 广义谱解释

设

\[
C_{\mathrm{ret},k-1}v_i=\rho_i C_{\mathrm{edit},k}v_i,
\quad v_i^\top C_{\mathrm{edit},k}v_j=\delta_{ij}.
\]

令 \(V=[v_1,\ldots,v_d]\)、\(u=V^\top g_k\)、\(\Delta=Vt\)。S6 变为

\[
\sum_i\left[u_i t_i+\frac12(1+\lambda\rho_i)t_i^2\right],
\]

因此

\[
t_i^\star=-\frac{u_i}{1+\lambda\rho_i},
\qquad
\mathcal A_k g_k=V^{-\top}
\operatorname{diag}\left(\frac1{1+\lambda\rho_i}\right)V^\top g_k.
\tag{S9}
\]

\(\rho_i\) 是保留曲率相对于编辑曲率的比值，而不是可直接观测的“实际编辑收益”。实际收益还受梯度方向、步长和优化器影响。

在此局部二次解处，保留代价为

\[
R_k(\Delta_k^\star)
=\frac12\sum_i\frac{\rho_i u_i^2}{(1+\lambda\rho_i)^2},
\qquad
\frac{dR_k}{d\lambda}
=-\sum_i\frac{\rho_i^2u_i^2}{(1+\lambda\rho_i)^3}\le0.
\tag{S10}
\]

该单调性仅针对固定曲率的二次最优解，不保证 Adam 的更新、真实 KL 或整个序列的最终遗忘也单调。

### 4.2 两种实际更新不能混为一谈

`relative` 在第 t 个优化步使用当前编辑梯度 \(g_{k,t}\)，计算 \(\mathcal A_k g_{k,t}\)，随后执行 Adam；轮内固定 \(\mathcal A_k\)。它复用了由局部模型导出的算子，但不是在每一步精确求解 S6。

`explicit` 直接对 S3 求梯度：

\[
\nabla J_k(\theta)=\nabla\mathcal L_k(\theta)
+\lambda C_{\mathrm{ret},k-1}(\theta-\theta_{k-1}),
\tag{S11}
\]

并交给普通 Adam。若没有遗留动量，本轮起点的正则梯度为零；`relative` 则从第一步就改变输入梯度。两者在有限步 Adam 下不等价。

## 5. K-FAC 块正则化及其梯度

对权重块 \(W_\ell\in\mathbb R^{m\times n}\)，令 \(\Delta W_\ell=W_\ell-W_{\ell,k-1}\)，采用列优先向量化：

\[
C_{\mathrm{ret},\ell}\approx A_{\mathrm{ret},\ell}\otimes B_{\mathrm{ret},\ell},
\quad A\in\mathbb R^{n\times n},\ B\in\mathbb R^{m\times m}.
\]

由 Kronecker 恒等式，

\[
R_k(\theta)=\frac12\sum_\ell
\operatorname{tr}\left(\Delta W_\ell^\top
B_{\mathrm{ret},\ell}\Delta W_\ell A_{\mathrm{ret},\ell}\right),
\tag{S12}
\]

在因子对称时有

\[
\nabla_{W_\ell}R_k=B_{\mathrm{ret},\ell}\Delta W_\ell A_{\mathrm{ret},\ell}.
\tag{S13}
\]

不需构造 \(mn\times mn\) 矩阵。每块二次项/梯度的密集乘法成本为 \(O(m^2n+mn^2)\)，另需 \(O(mn)\) 的锚点或位移存储。GPU 不足时可将轮次锚点放 CPU，但必须报告传输成本；不要逐步复制整个模型。

S12 对层求和，**没有默认再除以层数、参数数或 token 数**。若实现加入这些归一化，必须同步修改目标并重新解释 lambda。编辑损失的 token 平均方式也须固定。

主方法继续使用原论文 Eq. (17)：

\[
G^{\mathrm{adj}}=Q_B^{-\top}
\left[\frac{(Q_B^\top GQ_A)_{ij}}{1+\lambda b_i a_j}\right]_{ij}
Q_A^{-1}.
\tag{S14}
\]

能力因子替换为保留因子即可。两个因子分别阻尼所形成的曲率是 \((A_e+\delta_A I)\otimes(B_e+\delta_B I)\)，不等于简单 \(A_e\otimes B_e+\delta I\)；文稿应说明数值实现使用哪一种。

## 6. 历史状态：理想矩阵递推与实际因子递推

概念上，在第 k 轮完成后估计本轮数据的曲率 \(C_k^{\mathrm{post}}\)，将其加入保留状态：

\[
C_{\mathrm{ret},k}=\frac{M_{k-1}C_{\mathrm{ret},k-1}+T_k C_k^{\mathrm{post}}}{M_{k-1}+T_k},
\quad M_k=M_{k-1}+T_k.
\tag{S15}
\]

当前代码更接近分别更新因子：

\[
\bar A_k=\frac{M_{k-1}\bar A_{k-1}+T_k A_k^{\mathrm{post}}}{M_k},
\qquad
\bar B_k=\frac{M_{k-1}\bar B_{k-1}+T_k B_k^{\mathrm{post}}}{M_k},
\quad \widetilde C_{\mathrm{ret},k}=\bar A_k\otimes\bar B_k.
\tag{S16}
\]

S16 **不是** S15 的精确实现，因为

\[
(\sum_jw_j A_j)\otimes(\sum_jw_j B_j)
\ne\sum_jw_j(A_j\otimes B_j).
\]

两者相差跨来源项。S16 仍保持半正定，且只需固定数量的因子，但应明确称为“聚合因子的历史曲率代理”。

### 6.1 token 与 sample 权重

当前估计函数返回的第三项是有效 token 数。若因子是 token 平均值，第一版推荐 `retention_weight=tokens`，全程使用相同有效 token 掩码。

不能直接用请求数量覆盖 token 计数后，宣称得到了精确按请求平均的统计量。若要给每个请求相同权重，应先在每个请求内部归一化，再在请求之间平均；仅修改合并权重是另一种启发式。

`num_samples` 名称容易混淆；建议状态中明确保存 `num_tokens`、`num_requests` 和 `weight_unit`。出现零有效 token 时报告并跳过/拒绝该估计，不要用 `max(n,1)` 静默伪造一次观测。各层使用自己的有效计数。

### 6.2 固定和动态状态对照

- `fixed`：始终使用初始能力因子，不合并历史编辑。
- `post`：编辑结束后重新估计本轮因子，并用 S16 更新。
- 编辑前因子合并可作为后续消融，但不能代替 `post` 后仍宣称保护的是编辑完成时的局部几何。

### 6.3 理论边界

原始能力因子在 S16 中的系数为 \(M_0/(M_0+\sum_j T_j)\)，会随历史长度下降；这只是因子系数，不能直接当作完整 Fisher 的混合权重。长序列中应单独测量原始能力保持。

本方案采用移动锚点，无法保证相对于 \(\theta_0\) 的累计漂移有界，也不能恢复已遗忘的事实。若要保护各历史检查点，应重新定义多锚点目标并保留其线性项。

重复更新同一事实时，旧目标和新目标可能冲突；第一轮实验先使用无矛盾的编辑序列，之后再定义旧约束的替换/失效规则。

## 7. 拟议训练流程

```text
只用能力参考集初始化 retention，保存状态类型和有效 token 数
固定请求顺序并保存请求 ID；不读取未来编辑来估计统计量

for each arriving chunk D_k:
    explicit 模式：仅复制可编辑权重作为本轮固定锚点
    根据既定策略重置或继承 Adam 状态

    if mode == relative:
        在 theta_(k-1) 上估计当前 chunk 的 edit_factors
        将 retention 与 edit_factors 配对，建立本轮广义基
        使用 ProjectedAdam
    elif mode == explicit:
        固定本轮 retention 因子
        使用普通 Adam，不经过任何梯度过滤

    for the configured optimization steps/minibatches:
        edit_loss = 当前 mini-batch 的目标 token 交叉熵
        explicit: total_loss = edit_loss + lambda * quadratic_penalty
        relative: total_loss = edit_loss
        backward(total_loss)
        optimizer.step()
        记录实际参数增量、edit_loss、regularizer、total_loss

    if retention_update == post:
        在 theta_k 上重新估计本轮因子；禁止复用轮前缓存
        仅合并 detached A/B 和有效计数，丢弃旧广义基
    评估当前批、历史批与原始能力；保存状态元数据
```

新方法验证阶段建议各模式统一“每轮重置 Adam 状态”。已有结果复现则保留旧状态策略并显式记录，两者不要混在一张无说明的对照表中。

## 8. 逐文件修改清单（尚未实施）

### 8.1 `easyeditor/models/rhoedit/RhoEdit_hparams.py`

在 `AdamHyperParams` 的现有默认字段之后加入拟议字段；dataclass 非默认字段必须继续位于默认字段之前。

```python
# 建议接口，未实现
sequential_reg_mode: str = "relative"  # relative / explicit / none
retention_update: str = "post"         # fixed / post
retention_weight: str = "tokens"
curvature_estimator: str = "observed"  # observed / model_fisher
sequential_optimizer_state: str = "legacy"  # legacy / reset
seed: int = 69
seq_plot_every: int = 0
```

`observed`/`legacy` 默认值是为了显式承认并保留当前运行口径；与论文一致的新实验必须显式指定经过验证的 `model_fisher`，机制实验显式指定 `reset`。也可以把 estimator 改为新实验必填项，避免默认值被误用。

复用现有 `soft_lambda` 作为两模式的保护系数：relative 中进入谱滤波，explicit 中进入损失。无需先引入第二个 lambda。输出完整配置，避免超参数在动态属性赋值和 YAML 中分散。

### 8.2 `run_rhoedit.py`

修改 `get_arguments()`、`get_hparams()`、`calculate_model_name()` 和主入口：

1. 为上述配置提供同名 CLI 参数、choices 和检查，`get_hparams()` 负责唯一的配置合并。
2. 将种子初始化移动到参数解析之后；不要继续固定模块级 SEED 后仅记录新 seed。
3. 第一版新正则化模式只接入 `--sequential_edit --edit_cache_style sequential`；不与旧 `mix/disable` 路径交叉，避免重复或不一致更新。
4. 显式模式只使用普通 Adam；错误的模式组合应直接报错，不能静默回退。
5. 运行目录至少区分 mode、retention_update、estimator、state policy、seed、lambda、chunk 大小，并保存完整配置 JSON。
6. 接入已有 `SequentialEditTracker`：增加频率等所需参数，按其真实构造函数创建，并传给 `execute_sft_adam_sequential(..., tracker=tracker)`。
7. 如暴露 `factor_device`，把它同时传入 hparams 和优化器构造；当前 hparams/ProjectedAdam 有字段，但入口和 builder 未完整传递。

### 8.3 `utils.py` 的能力初始化和曲率适配层

位置：`_build_cov_cache_from_hparams()`、`calculate_cov_cache_with_old_data()`、`calculate_cov_cache_with_request()`、`estimate_chunk_kfac()`。

新增“只加载能力统计”的入口参数或独立函数。当前初始化还计算全 task_mom2_dataset 的因子，顺序分支随后只取能力 A/B；此额外计算在主路径中不必要，并容易让未来数据进入其他分支。真正的逐轮协议只在本轮到达后估计当前编辑因子。

将估计器选择集中在适配层，统一返回每层：

```python
{
    "A": detached_cpu_factor,
    "B": detached_cpu_factor,
    "num_tokens": valid_tokens,
    "num_requests": request_count,
    "estimator": estimator_name,
    "weight_unit": "tokens",
}
```

这是目标结构，当前代码还使用 `num_samples`；若先保留旧接口，应同步记录它的真实单位，不要只改一处键名。

轮前和轮后统计禁止共用与参数状态无关的缓存键。键/元数据应包括模型检查点、轮次、pre/post、请求 ID 哈希、估计器、采样种子、截断长度、token 掩码和精度。

### 8.4 `utils.py::execute_sft_adam_sequential()`

这是新增显式正则化的主要修改位置。

**A. 进入每轮、开始优化之前**

- `explicit` 模式仅对 `get_weights()` 返回的可编辑参数 `detach().clone()`，形成锚点快照；快照不参与梯度。
- 不要使用随权重刷新逻辑变化的 `current_weights_cpu` 充当正则锚点，应使用独立名称 `round_anchor`。
- 冻结本轮的 retention 因子；多 GPU 下按对应参数所在设备准备因子，不统一搬到 `model.device`。
- `relative` 模式继续估计当前 chunk 的编辑曲率并构建配对缓存。
- `explicit` 模式不需要当前编辑曲率来计算 S12，不应为了满足旧 builder 的 task_A/task_B 要求而额外加载未来编辑统计。

**B. 优化器创建**

- relative：复用 `build_optimizer_with_cov_caches()`，支持每轮 `reset` 时重新创建实例。
- explicit/none：直接创建普通 Adam，保持 lr、betas、eps、weight_decay 等与对照一致。
- 若选择 `legacy` 复现，记录 relative 的一阶动量再过滤行为；不要宣称它与普通 Adam 继承状态完全相同。

**C. 替换顺序函数中 `loss = outputs.loss` 附近的组织方式**

```python
# 结构示意，不是已修改代码
edit_loss = outputs.loss
if mode == "explicit":
    reg_loss = quadratic_retention_penalty(weights, round_anchor, retention)
    total_loss = edit_loss + hparams.soft_lambda * reg_loss
else:
    total_loss = edit_loss

total_loss.backward()
opt.step()
```

这里的 penalty 使用 S12。`W` 的梯度必须保留；只有 anchor、A、B detach。不要对 `W-anchor` 整体 detach，也不要使用 `.item()` 构造参与反传的损失。

原有 `if loss.item() >= 1e-2` 和 edit-loss 提前停止需要重新定义：显式模式下 edit_loss 很小并不表示正则项已优化好。最初对照建议采用统一且确定的实际 optimizer-step 预算，禁用旧的仅根据编辑损失跳步/停止；随后再独立验证提前停止策略。分别记录 edit_loss、reg_loss、lambda*reg_loss 和 total_loss。

**D. 每轮训练结束**

- `post` 分支在最新参数上重新估计本轮因子，再更新 retention；`fixed` 分支跳过合并。
- 回收本轮锚点和临时缓存，但不要删训练结果或历史评估数据。
- 若支持断点恢复，保存轮次、编辑顺序 ID、retention、计数、估计器、随机数状态及优化器策略；每轮边界 checkpoint 不需保存旧轮锚点，轮内恢复则必须保存。

### 8.5 拟新增 `easyeditor/models/rhoedit/sequential_regularization.py`

建议仅在未来实施时创建，集中管理：

- `snapshot_editable_weights(weights)`：轮次锚点。
- `quadratic_retention_penalty(weights, anchor, factors)`：S12 标量。
- 因子/参数名称映射、设备迁移和形状检查。

下面是单层公式的代码示意，保存在本文档中，不创建源文件：

```python
delta = weight.float() - anchor.to(device=weight.device, dtype=torch.float32)
A = factor_A.detach().to(device=weight.device, dtype=torch.float32)
B = factor_B.detach().to(device=weight.device, dtype=torch.float32)
A = 0.5 * (A + A.T)
B = 0.5 * (B + B.T)
assert A.shape == (delta.shape[1], delta.shape[1])
assert B.shape == (delta.shape[0], delta.shape[0])
layer_penalty = 0.5 * (delta * (B @ delta @ A)).sum()
```

循环对层求和；不建立完整 Kronecker 矩阵。浮点误差可能造成微小负值，应记录容差并检查因子，不能用 clamp 掩盖严重非 PSD 或转置错误。第一版采用 autograd，确认正确后才考虑手动添加 S13 梯度以节省图存储。

### 8.6 `utils.py::update_retention_eq13()` 和因子缓存

建议改名为 `update_retention_factors()`，避免错误指向当前论文 Eq. (13)。如果保留旧名作为兼容包装，需要注释其来源，而不是继续混用论文编号。

合并仅处理 detached A/B 与有效计数，不混入上一轮 `soft_q_*`、`soft_eig_*` 或 task 因子。新一轮必须重新生成匹配当前编辑因子的广义基。保留因子只需对称和 PSD；没有必要为其随意加入不同于方法定义的强阻尼。

### 8.7 `projected_adam.py`

`_soft_kfac_precondition()` 的核心变换已经对应 S14，第一版无需改写。

需要做的是规范优化器生命周期：

- `reset`：每轮重建优化器，所有方法共同采用这一设置用于机制对照。
- `legacy`：维持现有 `reset_cache()` 一阶动量再过滤，作为旧版本复现设置。
- 后续如引入“不改变动量的 carry”，应单独命名并实验，不伪装成现有行为。

不能在 ProjectedAdam.step 中偷偷加入显式正则梯度，再把它归为纯 explicit 方法。

### 8.8 `easyeditor/models/rome/layer_stats.py`

当前 observed 路径与论文模型采样 Fisher 不一致；已有 `layer_stats_kfac_fisher_with_txt_tgt()` 可以作为实现起点，但不是直接替换函数名即可：其签名、数据路径、截断和掩码都不同，能力语料还需要对应估计入口。

建议统一以下约定并验证：

1. 采样分布使用模型原始预测，即 temperature=1；否则估计的是另一分布的几何。
2. 能力侧与编辑侧都显式选择 estimator；模型采样模式中，编辑损失依旧用真实编辑目标。
3. 标签采样只发生在曲率估计，不改写训练目标。
4. 规定有效位置：编辑侧采用目标 token 或完整文本必须二选一并记入配置；能力侧使用有效参考 token。各估计器对照使用相同位置掩码。
5. 对齐 next-token shift；不能把所有 token ID=0 都默认视作 padding，应依据 tokenizer 和 attention_mask。
6. 估计过程使用一致的 eval/dropout 约定，并在 finally 中恢复训练模式、requires_grad 状态和移除 hooks。
7. 记录 MC 样本数、种子与 token 归一化；冻结统计量计算图。
8. 先用小型分类模型验证采样梯度外积在重复采样下接近解析 model Fisher，再运行 LLM；不能仅因函数名含 fisher 就认定实现符合论文。

### 8.9 `seq_tracker.py`、配置和结果命名

- 入口接通已有 tracker，并明确其中准确率为 teacher-forced token proxy。
- 增加每轮正则量、相对轮次锚点的位移范数、实际步数，以及相对原模型的能力变化。
- 当前/历史事实的生成式评估另行接入；保存检查点时机和固定评估子集 ID，避免将所有轮次都当作独立测试集调参。
- 时间分别记录估计、分解、更新和评估；GPU 计时需正确同步，并报告所有训练设备的显存。
- 编辑器状态统计包括保留因子、当前编辑因子、广义基和显式锚点；不要只统计一个 retention 字典就声称是总内存。
- YAML 先增加配置说明，CLI 覆盖规则唯一。输出 JSON 保存完整配置，结果目录避免模式之间互相覆盖。

## 9. 未来实现后的验证顺序

这些是应当编写/运行的验证，不是本次已执行的测试。

1. 小矩阵核查 S12 与显式 Kronecker 二次型相等。PyTorch 默认 flatten 是行优先；列优先验证要使用 `delta.T.contiguous().reshape(-1)`，否则会混淆 A/B 顺序。
2. autograd 梯度与 \(B\Delta W A\) 比较；检查 \(\Delta W=0\) 时正则和梯度均为零，PSD 因子下正则非负。
3. 锚点在同一轮内保持固定，在下一轮更新；参数能收到梯度，因子与锚点没有梯度。
4. 验证 lambda=0 时，两模式在相同 Adam 状态、批次和随机性下还原无保护更新。估计器若消耗随机数，应使用独立 RNG 或固定轨迹，避免污染比较。
5. 验证 explicit 路径不会进入 ProjectedAdam，fixed 不合并历史，post 不复用 pre 缓存。
6. 合并两个不同 token 数的人工因子，与手工加权结果比较；拒绝零 token 并测试层间计数一致性。
7. 使用 2–3 个小 chunk 做烟雾运行，确认可以完成训练、恢复 checkpoint、输出历史评估矩阵。
8. 正式做 3,000 条/100 条每轮的机制实验；固定事实、顺序和预算，对比多个种子。再扩大序列和模型。

优先指标：当前生成成功率、历史生成成功率矩阵、相对各批初次完成时的遗忘、原始能力分任务变化、局部性、实际参数位移和成本。仅有最终整体平均值不足以验证顺序保护。

## 10. 拟议命令（当前不能运行）

下列新增参数需按上述指南实现后才能使用，展示的是期望的实验接口：

```bash
python run_rhoedit.py --model llama3-8b --data_type zsre \
  --sequential_edit --num_edits 100 --edit_cache_style sequential \
  --sequential_reg_mode relative --retention_update post \
  --retention_weight tokens --curvature_estimator model_fisher \
  --sequential_optimizer_state reset --soft_lambda 0.1 \
  --seed 69 --seq_plot_every 1 --plat_name none --no_wandb

python run_rhoedit.py --model llama3-8b --data_type zsre \
  --sequential_edit --num_edits 100 --edit_cache_style sequential \
  --sequential_reg_mode explicit --retention_update post \
  --retention_weight tokens --curvature_estimator model_fisher \
  --sequential_optimizer_state reset --soft_lambda 0.1 \
  --seed 69 --seq_plot_every 1 --plat_name none --no_wandb
```

0.1 仅为命令示例，不是已验证的推荐最优值。相同 lambda 不保证两种 Adam 实现具有相同实际保护强度，正式比较需要相同调参预算的权衡曲线。

## 11. 可用于新增论文小节的文字

> We extend RhoEdit to sequential editing by maintaining a curvature-based retention state. At round k, a quadratic penalty measures parameter displacement from the round-start model in the geometry of retained behavior. A local second-order model of the current edit loss yields the same relative-curvature adjustment as in the single-stage formulation, with capability curvature replaced by the retention state. After completing the round, we estimate curvature on the newly edited examples and incorporate its factors into the state used by subsequent rounds. This update is a factorized approximation to historical geometry, rather than an exact sum of Fisher matrices. Our practical algorithm applies the resulting gradient adjustment through Adam; the local quadratic analysis does not imply a global no-forgetting guarantee.

补充说明显式正则化对照：

> To distinguish the effect of the retention state from the choice of update rule, we also optimize the edit loss with an explicit K-FAC quadratic penalty using ordinary Adam. Both variants use the same data and retention-update protocol, while differing in how the retention geometry enters optimization.

在得到结果之前，只写方法定义与待检验假设，不写已减少遗忘、跨模型一致领先或保持所有能力等结论。

## 12. 推荐实施顺序

1. 固定和核查曲率估计器、掩码、计数、缓存来源，记录原实验版本。
2. 增加配置并接通现有 tracker，保存编辑顺序与有效优化步数。
3. 新增 explicit 普通 Adam 分支、固定轮次锚点和 S12 正则项。
4. 规范 relative/fixed/post 分支与优化器状态，保留可追溯的旧实现模式。
5. 完成小矩阵和多轮烟雾验证，再开展正式对照。
6. 将本推导整理为论文 Sec. 3.4 / Algorithm 2，原有单阶段推导与结果继续保留其适用范围。
