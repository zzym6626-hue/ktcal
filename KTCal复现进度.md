# KTCal 复现进度

## 项目概况
在 TDTracker（CVPR 2025 事件相机眼动追踪第三名方案）代码基础上，复现论文 **KTCal：闭环预测-校正跟踪框架**（含 STNet 轻量时空网络）。

- 数据集：SEET（train 320 序列 / test 40 序列，每序列 100 帧，事件帧 2×60×80）
- 环境：macOS CPU（PyTorch 2.1.0，无 CUDA）

## 已完成模块

- `models/STNet.py` — 解耦时空卷积(ITD) + Bi-GRU(ETD) + FC 回归头，约 2.08M 参数（论文 2.28M）
- `models/kalman_filter.py` — 匀速卡尔曼滤波器 [x,y,vx,vy]，含阈值触发重置（W3）
- `models/ncc_matcher.py` — 特征提取器(32ch残差) + 逐窗口归一化互相关，Soft/Hard-Argmax（W4）
- `ktcal_tracker.py` — 闭环三层阈值调度 + 回溯校正 + 真实延迟测量
- `train_ktcal.py` — 联合损失 `L_total = L_ST + 0.3·L_match`（跨帧匹配）
- `test_ktcal.py` — 闭环评估（精度 + tier 分布 + 延迟）

## 论文核心机制对应

- 三层触发（W1）：ε<3px 用卡尔曼；3≤ε<5px 异步 STNet；ε≥5px 强制 STNet
- 回溯校正（W1/W3）：σ>1px 重置滤波器，协方差 diag([10,10,10,10])
- 模板动态更新：仅 STNet 触发时用其精确结果刷新模板（抗漂移）
- 联合训练（W4）：STNet + NCC 端到端，仅用真值坐标监督

## 遇到的问题及修复

1. **CUDA 依赖** — checkpoint/metrics 硬编码 `.cuda()`，Mac 无 GPU 崩溃 → 加 `map_location='cpu'` + `--use_cpu` 支持
2. **STNet 每帧都跑** — 初版预计算全帧，违背"按需触发" → 改滑动窗口按需推理 + 真实测延迟
3. **NCC 坐标映射 bug** — 响应图→图像坐标漏模板半宽偏移，ε 恒~14px、100% tier3 → 统一坐标系
4. **模板漂移** — 每帧用卡尔曼位置更新导致累积误差 → 仅 STNet 结果刷新模板
5. **训练/推理不一致** — match_loss 同帧裁剪 → 改跨帧训练（模板取 t-gap 帧）
6. **裁剪越界崩溃** — 卡尔曼预测发散出界 → 中心钳制到 [0,1]
7. **NCC 置信度过低（关键）** — conf 中位数仅 0.37、>0.5 仅 14%
   → 根因：`ncc_response` 用整个搜索区域范数做分母（错），局部窗口范数被压低
   → 修复为**逐窗口归一化的真正 NCC**：`NCC=<S_win,T̂>/|S_win-mean(S_win)|`
   → 修复后 conf 中位数 0.61，**>0.5 达 62-71%**

## 当前指标

STNet 单独验证：**MSE ≈ 2.00px**（论文 SEET 1.46px；仅训 ~23 epoch，CPU 两次均被 OOM 提前终止）

闭环评估（NCC 归一化修复前后对比）：

- 修复前：MSE 2.75px，tier1 10.4%，STNet 触发 89.5%，延迟 78.8ms
- **修复后：MSE 2.82px，tier1 42.1%，tier2 2.3%，tier3 55.6%，STNet 触发 57.9%，延迟 50.9ms**
- 精度：p3=65% / p5=90% / p10=98%

## 延迟优化历程（P0）

| 阶段 | NCC单次 | 关键路径延迟 | baseline(STNet每帧) | STNet触发率 |
|------|--------|------------|--------------------|-----------|
| 加强NCC + window100（串行） | ~22ms | 50.9ms | 84.78ms | 58% |
| 加强NCC + window16（串行） | ~22ms | 29.5ms | 12.65ms | 58% |
| 轻量NCC + window16（串行） | ~1ms | 14.06ms | 12.65ms | 54% |
| **轻量NCC + 并行模型** | ~1ms | **1.63ms** | 12.65ms | 56% |

关键修复：
- 双向 GRU 无法真正增量 → STNet 用有界窗口(16)，per-frame O(W)
- NCC 特征提取器退回轻量 2 层 16ch，单次 22ms→1ms
- **并行/异步延迟模型（论文核心）**：三路径并发，STNet 异步执行不阻塞主路径，
  关键路径 = 卡尔曼 + NCC。关键路径 1.63ms vs baseline 12.65ms，**降幅 87%**
  （论文 4.7→1.17ms 降 75%，量级一致）

## 结论

- 闭环并行加速机制验证成功：**关键路径 1.63ms（↓87%），与论文量级一致**
- 你的核心判断（模板匹配可并行、三路径同时进行、这是论文最厉害处）全部验证正确
- 代价（延迟↔精度权衡）：并行下 tier3 先输出未经 STNet 校正的卡尔曼预测，
  MSE 升到 4.29px、卡尔曼重置暴增到 1105 次

## P3 修复（重置过多，已改代码，尚未重测）

- 根因：tier3 原用「update 后残差 vs stnet」判重置，且 τ3=1px 过严，
  叠加 tier3 占比高 → 每帧都在重置
- 修复：重置判据改为「卡尔曼**预测** kf_pred vs STNet 全局定位」（真实漂移检测），
  τ3 放宽 1px→3px；未漂移时正常 update，仅真漂移才 reset
- 预期：重置次数大幅下降、精度回升（待重测验证）

## 已知差距与待办

- 精度未达论文（论文 MSE 1.46 / STNet触发~20%）；触发率仍 ~56%
- 根本瓶颈：事件帧稀疏使 NCC 匹配 ε 卡 3px 边界 + CPU 训练两次 OOM 未跑满
- [ ] 重测 P3 效果（重置次数、MSE）
- [ ] 训练防 OOM 跑满 30+ epoch（已改 batch=8）
- [ ] 可选：扩大模板/搜索窗 15/31（当前仍 11/21）

## 最终结果（500 轮训练早停@69 + 后处理）

- 训练：500 轮上限 + 早停(patience=20)，实际在 epoch 69 停，STNet MSE **1.72px**（30 轮时 2.0）
- 后处理：Savitzky-Golay 轨迹平滑（w=15, poly=3），加在 `test_ktcal.py`

| 指标 | 30轮 | 500轮(早停@69) | +后处理 | 论文 |
|------|------|--------------|--------|------|
| STNet MSE | 2.0 | 1.72 | — | 1.46 |
| 闭环 MSE | 4.32 | 4.16 | **3.81** | 1.46 |
| p5 精度 | — | 72.4% | **76.7%** | — |
| p10 精度 | — | 92.6% | **94.8%** | — |
| 关键路径延迟 | 1.5ms | 1.49ms | 1.49ms | 1.06ms |
| 卡尔曼重置 | 1105 | 0 | 0 | — |

- 后处理有效：闭环 MSE 4.16→3.81（↓8.5%），降幅超 TDTracker 后处理(1.55→1.49)
  ——并行异步轨迹抖动更大，平滑收益更多
- 原 1.49ms 为排除 STNet 后的记账快路径，不是真实异步 wall-clock；当前顺序实现实测为 14.92ms/帧。
- 精度仍差 ~2.6×（3.81 vs 1.46），受触发率偏高(56%)、时序窗口训练/推理不一致和闭环实现缺口影响。

## 2026-07-30 P0 实验可信度修复

- 训练改为从 320 条 train 序列按 seed=42 固定拆分 288/32，test 集不再用于早停或 checkpoint 选择。
- 默认 CPU batch=8；checkpoint 新增 epoch、val 指标、seed、配置与拆分索引，并兼容旧权重读取。
- 评估可通过 `--output_json` 保存结构化结果，且区分平均像素欧氏误差与平均像素平方误差。
- 完整回归通过：离线 1.6467px；闭环 4.1615px，后处理 3.8055px；真实顺序 wall-clock 14.92ms/帧。
- 现有权重曾使用 test 集选模，只保留为开发回归基线；下一步必须用新划分从头训练，再进入回溯闭环和真实异步实现。

## 2026-07-30 真实异步闭环实现

实现内容：
- `ktcal_tracker.py` 新增单后台 STNet worker，任务携带 `frame_id`、窗口快照和提交时间；同长度窗口最多 4 个组成 batch。
- CPU 使用独立线程；CUDA 使用独立 stream 和 CUDA event 计时，不用全设备同步破坏并行。
- tier1 仅 Kalman+NCC；tier2 提交后台 STNet 并继续处理后续帧；tier3/NCC 低置信按论文要求同步 STNet，防止等待期间 Kalman 发散。
- 后台结果只在主线程回收并按帧重放 Kalman，避免并发修改滤波器和模板；旧结果不能覆盖更新帧的模板。
- 默认继续采用已验证更稳的 STNet 高置信融合；`--hard_reset` 可用于 τ3 硬重置对照。
- `test_ktcal.py` 分别报告即时在线轨迹、闭环校正轨迹、异步计算量、tier3 同步计算、真实 wall-clock 和结果晚到帧数。

使用旧开发权重、macOS CPU、暂停 clean 训练后测试 40×100 帧：

| 指标 | 结果 |
|------|------|
| 即时在线平均像素误差 | 3.5309 px |
| 闭环校正平均像素误差 | 3.4755 px |
| + Savitzky-Golay | **3.2825 px** |
| p5（闭环 / 后处理） | 82.05% / 84.38% |
| 在线关键路径 | 16.7720 ms/帧 |
| tier2 异步 STNet 计算量 | 1.4743 ms/帧 |
| tier3 同步 STNet 计算量 | 14.5428 ms/帧 |
| 真实并发 wall-clock | **16.8318 ms/帧** |
| 异步结果晚到 | 4.90 帧 / 59.35 ms |
| tier1 / tier2 / tier3 | 43.1% / 3.9% / 53.0% |
| STNet 总触发率 | 56.9% |

关键结论：
- 真异步确实工作，但只有 tier2 的 3.9% 帧可隐藏 STNet；tier3 占 53.0%，同步 STNet 决定了总延迟。
- 精度较旧闭环后处理 3.8055px 提升到 3.2825px，但 CPU wall-clock 从 14.92 增至 16.83ms/帧。
- 尝试把所有 tier3 也异步会因结果平均晚到约 36-38 帧导致在线 Kalman 发散，最终误差 15-16px，已放弃。
- 下一步降低延迟的首要目标仍是把 tier3/总触发率降到论文约 20%，而不是继续增加线程。
- 上述结果仍使用旧开发权重；`checkpoint_ktcal_clean` 早停完成后必须重新做独立 test，得到无泄漏最终指标。

## 2026-07-30 tier3 根因诊断

在不修改模型、阈值和输出的前提下，对完整 4000 帧增加 tier3 原因、NCC 峰值边界和 GT 只读归因：

- tier3 2123 帧：首帧 40、NCC 低置信 1116、`epsilon>=5px` 967。
- 置信度分布：`<0.3` 839、`0.3-0.4` 95、`0.4-0.5` 182、`0.5-0.6` 558、`>=0.6` 2286。
- 可靠置信下的 epsilon：`<3` 1726、`3-5` 151、`5-7` 305、`7-10` 662。
- 全部 NCC 边界峰 1589；低置信 tier3 中 625 个边界峰；大误差 tier3 中 **956/967（98.9%）**为边界峰。
- GT 归因：低置信帧上 Kalman 3.56px、NCC 5.98px，NCC 仅 23.5% 更好；大误差帧上 Kalman 5.45px、NCC 9.98px，NCC 仅 11.6% 更好。

结论：当前主因是 NCC 假边界匹配，不是单纯阈值过严。只将 `ncc_thresh` 从 0.5 降至 0.4 最多搬走 182 帧，将 `tau2` 从 5 放至 7 最多搬走 305 帧，合计也只能将 tier3 约从 53.1% 降至 40.9%，且会接纳更多比 Kalman 更差的 NCC。不建议直接放宽阈值。

### `search_size=31` 对照（不采纳）

`test_ktcal.py` 已增加 `--template_size/--search_size`，默认仍为 11/21。完整测试 `--template_size 11 --search_size 31`：

| 指标 | 11/21 | 11/31 |
|------|------:|------:|
| 后处理平均像素误差 | **3.2764px** | 3.6141px |
| tier3 | **53.1%** | 54.7% |
| STNet 触发率 | **56.9%** | 58.7% |
| 全部边界峰 | 1589 | **1513** |
| 大误差边界峰 | **956** | 1033 |
| `epsilon>=10px` | **0** | 1049 |
| 真实 wall-clock | **15.79ms/帧** | 17.14ms/帧 |

扩大搜索窗虽略减边界峰总数，却引入大量远处假高峰，使 `epsilon>=10px` 从 0 增至 1049，精度、触发率和延迟全部变差，因此不修改默认值。下一步应先给 NCC 响应加入以 Kalman 预测为中心的距离先验/边界惩罚，再测试较温和的 25 或 31 搜索窗；不能只扩大窗口。

## 2026-07-30 clean 权重最终评估

训练使用 seed=42 固定拆分 288 train / 32 val，epoch 70 早停，最佳 checkpoint 为 epoch 50，val 平均像素误差 1.4641px。正式文件位于 `checkpoint_ktcal_clean/`。

| 模式 | 平均像素误差 | 后处理 | p5（后处理） | STNet触发 | wall-clock |
|------|-------------:|-------:|---------------:|----------:|-----------:|
| STNet batch | **1.6997px** | — | 98.83% | 100% | 1.65ms/帧 |
| 异步闭环 soft fusion | 3.7027px | 3.4416px | 79.83% | 58.7% | 18.83ms/帧 |
| hard reset, `tau3=1` | **2.9679px** | **2.7375px** | **89.25%** | 60.6% | 23.56ms/帧 |
| hard reset, `tau3=1.5` | 2.9851px | 2.7457px | 89.18% | 60.2% | 28.98ms/帧 |
| hard reset, `tau3=3` | 3.0273px | 2.7679px | 88.75% | 60.0% | 28.06ms/帧 |

结论：clean 离线 STNet 比旧受污染权重的 1.6467px 回落到 1.6997px，这是可信泛化成绩。hard reset 明显提升闭环精度，但 `tau3=1/1.5/3` 的重置次数仍有 1906/1654/1210（47.7%/41.4%/30.3%），无法复现论文 14.8%。继续扫阈值收益有限，优先解决 STNet 时序窗口分布差和 NCC 边界假峰。

## 2026-07-30 NCC 搜索窗与运动先验对照（不采纳）

`ktcal_tracker.py` 新增可选 NCC 重打分：`adjusted=NCC-lambda*distance^2-boundary_penalty`。定位使用调整分数，但 confidence 仍读取原始 NCC；`test_ktcal.py` 新增 `--ncc_center_penalty/--ncc_boundary_penalty`，默认均为 0，不改变正式基线。

以下机制对照使用旧开发权重，目的是判断 NCC 假峰来源；正式精度仍以上一节 clean 权重为准：

| 配置 | 后处理误差 | tier3 | 边界峰 | `epsilon>=10` | STNet触发率 |
|------|-----------:|------:|-------:|--------------:|------------:|
| **11/21，无先验** | **3.2764px** | 53.1% | 1589 | **0** | **56.9%** |
| 11/25，无先验 | 3.3952px | 53.6% | 1514 | 520 | 57.4% |
| 11/31，无先验 | 3.6141px | 54.7% | 1513 | 1049 | 58.7% |
| 11/31，lambda=0.05，边界=0.02 | 3.6446px | 54.0% | 1078 | 1025 | 57.8% |
| 11/31，lambda=0.20，边界=0.05 | 6.8073px | **27.7%** | **44** | 37 | **31.2%** |

强先验能显著降低边界峰和触发率，但会把 NCC 锁在 Kalman 中心附近；Kalman 偏移后形成错误自确认，精度严重下降。弱先验不足以压制远处假峰，25/31 搜索窗也均不如 21。结论是保持默认 11/21 和零先验；后续应改进/重训 NCC 特征与匹配损失，并使用 clean 权重重新验证，不能依靠扩大窗口或固定运动先验。

## 2026-07-30 NCC hard-negative 专项训练

`train_ktcal.py` 新增 `--ncc_only`，冻结/跳过 STNet，仅微调 NCC；验证和早停改用固定采样的 NCC hard-argmax 像素误差，不再使用无关的 STNet 指标。新增损失：

- 原跨帧 soft-argmax 坐标损失；
- 全响应 hard-negative margin：正确峰需高于正样本邻域外最强假峰；
- 边界 hard-negative margin：正确峰需高于响应图边界最强峰；
- 离散峰值交叉熵：直接对齐推理阶段 hard argmax。

正式训练从 `checkpoint_ktcal_clean/ncc_best.pth` 初始化，固定 seed=42 和 288/32 划分，输出到 `checkpoint_ktcal_ncc_hardneg/`，不覆盖 clean 基线。配置：batch=8、frames/seq=8、lr=5e-4、epoch上限100、patience=15、margin=0.15、boundary margin=0.10、loss权重=1.0/0.5/0.2。后台任务 ID `9wpfzv`。

训练在 epoch 23 早停，最佳 epoch 8，validation NCC hard-argmax error 3.2509px、边界率 47.27%。使用 clean STNet、新 NCC、`--hard_reset --tau3 1` 独立测试：

| 指标 | clean NCC 基线 | hard-negative NCC |
|------|---------------:|------------------:|
| 闭环平均像素误差 | **2.9679px** | 3.0332px |
| 后处理误差 | **2.7375px** | 2.7916px |
| p5（后处理） | **89.25%** | 87.52% |
| tier3 | 56.0% | **54.2%** |
| STNet 触发率 | 60.6% | **57.7%** |
| 边界峰 | 1661 | **1515** |
| hard reset | 1906 | **1783** |

新 NCC 将边界峰降低 8.8%、STNet 触发率降低 2.9 个百分点，但后处理误差增加 0.0541px（约2.0%），因此不替换 clean NCC。说明局部 NCC hard-argmax validation 与最终闭环精度不完全一致，当前 margin/CE 权重压制假峰的同时损伤了有效匹配。候选权重和测试结果保留在 `checkpoint_ktcal_ncc_hardneg/`。

### NCC 闭环 validation 选模（不替换 clean）

`train_ktcal.py --ncc_closed_loop_val` 使用 clean STNet + hard reset 在固定 validation 序列上直接运行部署闭环，选模分数为 `闭环平均像素误差 + 1.0*STNet触发率`。训练开始前先保存 clean NCC epoch 0 基线。低权重训练在 epoch17 早停，保存 epoch9；8条 validation score 从3.9487改善到3.9022。

独立 test 中候选后处理误差2.7411px、触发率59.7%，clean NCC为2.7375px、60.6%，没有形成可靠精度提升。全部32条 validation比较：clean/hard-negative/闭环选模 score分别为4.2024/4.1558/4.1646，两版微调在validation小幅改善但在test均未泛化，因此停止继续调NCC权重，正式基线继续使用clean NCC。

## 2026-07-30 STNet 随机短窗口微调（闭环新基线）

当前闭环每次只向STNet输入最近16帧，而clean STNet训练只监督100帧序列，存在明显窗口分布差。`train_ktcal.py --stnet_only` 新增4/8/16随机短窗训练：每个batch固定执行短窗损失，25%的batch同时保留完整100帧损失；validation始终同时计算完整序列与固定随机短窗误差，并按两者平均选模。NCC不创建、不更新。

正式微调从 `checkpoint_ktcal_clean/stnet_best.pth` 初始化，lr=5e-5、batch=8、patience=6，在epoch10早停并保存epoch4。相同固定validation采样：clean完整/短窗误差1.4603/3.3933px，候选1.4969/3.1553px，短窗改善7.0%、完整退化2.5%。

独立test结果：

| 指标 | clean STNet | 短窗 STNet |
|------|------------:|-----------:|
| batch离线误差 | **1.6997px** | 1.7145px |
| 闭环即时误差 | 3.0393px | **2.8412px** |
| 闭环最终误差 | 2.9699px | **2.7851px** |
| 后处理误差 | 2.7386px | **2.5627px** |
| p5（后处理） | 89.28% | **91.40%** |
| STNet触发率 | 60.28% | **59.68%** |
| hard reset | 1897 | **1846** |
| 同环境wall-clock | 25.7839ms/帧 | **25.1213ms/帧** |

短窗STNet将闭环后处理误差改善6.42%、wall-clock降低2.57%，代价是batch离线误差退化0.87%。在该阶段，`checkpoint_ktcal_clean/stnet_best.pth` 作为batch离线最佳，`checkpoint_ktcal_stnet_short/stnet_best.pth` 作为闭环STNet；后续已被最后一步增强模型超过。该阶段结果保存在 `checkpoint_ktcal_stnet_short/test_batch.json`、`test_closed.json` 和同环境clean对照 `test_clean_same_environment.json`。

## 2026-07-30 真实闭环 rollout NCC 重训（不替换 clean）

新增 `collect_ncc_rollout.py`、`train_ncc_rollout.py` 和 `select_ncc_rollout.py`。在固定 288/32 train/val split 上，使用正式短窗 STNet、clean NCC 和 hard reset 闭环，缓存真实历史模板、Kalman search center、NCC 峰、tier、置信度、边界标记和 GT。缓存位于 `checkpoint_ktcal_ncc_rollout/`。

| rollout统计 | train | val |
|------------|------:|----:|
| 样本数 | 28,512 | 3,168 |
| GT在11x11响应范围内 | 75.8% | 75.9% |
| GT在响应范围外 | **24.2%** | **24.1%** |
| NCC边界峰 | 41.0% | 38.8% |
| 低置信tier3 | 28.2% | 23.4% |
| 大误差tier3 | 24.3% | 25.2% |

val 的 798 个大误差 tier3 中有 458 个（57.4%）GT 已落在响应范围外。说明大量边界峰并非仅靠特征训练可解；11/21 固定局部搜索本身构成约24%的几何上限。

训练两组 deployment-aware NCC：B 组使用真实 Kalman center + GT template + 可匹配坐标监督；D 组使用真实 Kalman center + 实际历史 template，并对失败、边界峰和陈旧模板重采样，对范围外样本只做低置信监督。完整32条val闭环选择结果：

| 模型 | val最终误差 | val后处理 | 触发率 | 边界率 |
|------|-----------:|----------:|-------:|-------:|
| clean NCC | 3.4418px | 3.2441px | 52.13% | 38.50% |
| B组 epoch12 | **3.3743px** | **3.1991px** | 51.31% | 36.47% |
| D组 epoch4 | 3.3993px | 3.2164px | **51.25%** | **36.06%** |

按 val 预先选择 B 组作为唯一 test 候选，D 不接触 test。独立 test 对比正式短窗 STNet + clean NCC：

| 指标 | clean NCC | rollout B NCC |
|------|----------:|--------------:|
| 即时误差 | **2.8412px** | 2.8424px |
| 最终误差 | **2.7851px** | 2.8063px |
| 后处理误差 | **2.5627px** | 2.6143px |
| p5后处理 | **91.40%** | 90.85% |
| STNet触发率 | 59.68% | **58.50%** |
| 边界峰 | 1673 | **1574** |
| hard reset | 1846 | **1796** |

候选边界峰下降5.9%、触发率下降1.18个百分点，但后处理误差增加0.0516px，且触发改善未达到预设2个百分点门槛，因此拒绝候选，正式NCC继续使用 `checkpoint_ktcal_clean/ncc_best.pth`。候选与结果保留在 `checkpoint_ktcal_ncc_rollout_gt/`、`checkpoint_ktcal_ncc_rollout_historical/`。

结论：真实失败回灌再次出现“压低边界峰但损伤有效匹配”的权衡。下一步不应继续在固定11/21响应图上加 loss；要突破约24%几何不可匹配，应改善 Kalman/STNet 预测中心，或设计由低置信触发的受控二阶段扩大搜索，而不能对所有帧固定扩大窗口。

## 2026-07-30 Kalman 与受控搜索实验（正式配置不变）

val GT 轨迹的理想常速度一步外推平均误差仅0.027px、无一帧超过3px，说明24%几何越界来自观测/状态污染，不是轨迹运动过快。为此参数化 `reset_velocity_mode`、`low_conf_policy`、Kalman Q/R和`velocity_decay`，默认值全部保持旧行为。

Reset速度策略的32条val结果：legacy/前一帧差/清零/保留旧速度的最终误差分别为3.4500/3.5183/3.5857/**3.4311px**，触发率51.97/51.19/49.78/**51.16%**。按val选择“保留旧速度”进入test：即时2.8245px、最终2.7890px、后处理2.5717px、触发58.10%；相对正式2.8412/2.7851/2.5627px、59.68%只改善即时与触发，最终/后处理略差，因此不替换。

低置信策略实验：不用STNet直接Kalman预测会发散到563px；保持位置也有6.11px；全部异步STNet最终val约3.42px，但在线误差225px且触发89%；保持位置+异步STNet在线4.57px、最终3.43px、触发74.9%。证明当前低置信同步STNet是必要的安全路径，不能为追求低同步率直接移出关键路径。

速度阻尼 `0/0.25/0.5/0.75/1.0` 中val因果最终误差以`velocity_decay=0`最佳（3.4274px，触发50.94%）；增大Q、减小R及与preserve组合均无进一步因果收益。独立test：即时2.8312px、最终2.7857px、后处理2.5835px、触发57.78%、reset1716。触发下降1.90个百分点但后处理恶化0.0208px，仍不替换正式配置。结果见 `checkpoint_ktcal_kalman_reset_fix/`、`checkpoint_ktcal_low_conf_policy/`、`checkpoint_ktcal_kalman_tuning/`。

新增 `analyze_ncc_fallback.py` 离线评估31x31二阶段搜索。val中1539个低置信/边界候选的大窗平均误差13.91px，差于局部NCC 9.80px和Kalman 7.89px。阈值0.6且置信增益0.05只接受45帧（总帧1.42%），其中大窗优于局部55.6%、优于Kalman71.1%，但候选距Kalman平均8.76px，仍会进入tier3。可安全接受的子集过小且误接风险高，因此不接入闭环。分析见 `checkpoint_ktcal_ncc_fallback/val_search31_analysis.json`。

本轮结论：Kalman阻尼、reset策略、低置信调度和受控大窗均无法同时超过当时正式精度与触发率，因此保持legacy reset和低置信同步STNet。随后已继续完成STNet最后一步监督、L2/增强及2.28M结构实验，结果见下节。

## 2026-07-30 STNet最后一步语义与增强（新的闭环正式最优）

训练路径新增最后一步监督、归一化坐标L2、1/2/4/8/16冷启动窗口，以及标签同步水平翻转、旋转、平移和event dropout；默认全部关闭以兼容旧命令。模型均使用固定288/32 split，独立test只评估完整val预选出的增强组。

| val闭环模型 | 在线误差 | 最终误差 | 后处理 | 触发率 |
|-------------|---------:|---------:|-------:|-------:|
| 原短窗STNet | 3.5072px | 3.4500px | 3.2515px | 51.97% |
| 最后一步KL | 3.4914px | 3.4192px | 3.2520px | 52.34% |
| 最后一步KL+L2 | 3.4564px | 3.4163px | 3.2403px | 52.41% |
| **KL+L2+增强** | **3.4032px** | **3.3369px** | **3.1439px** | 52.16% |
| 2.282M第三层7x7 | 3.5375px | 3.5009px | 3.3219px | 51.91% |

增强配置：flip=0.5、rotation=±5°、shift=±2px、event dropout=3%，最后一步KL + 100×归一化坐标L2，窗口1/2/4/8/16，25% batch保留100帧最后步监督。权重位于 `checkpoint_ktcal_stnet_laststep_aug/stnet_best.pth`。

独立test对比：

| 指标 | 原短窗正式 | 新增强STNet |
|------|-----------:|------------:|
| 即时误差 | 2.8412px | **2.7855px** |
| 最终误差 | 2.7851px | **2.7211px** |
| 后处理误差 | 2.5627px | **2.4908px** |
| p5后处理 | 91.40% | **92.58%** |
| STNet触发率 | 59.68% | **58.70%** |
| hard reset | 1846 | **1816** |

新模型同时改善在线、最终、后处理、p5和触发率，正式闭环STNet更新为 `checkpoint_ktcal_stnet_laststep_aug/stnet_best.pth`；NCC仍使用 `checkpoint_ktcal_clean/ncc_best.pth`，参数仍为`--hard_reset --tau3 1`。新模型batch离线误差1.7854px，弱于clean STNet的1.6997px，因此离线batch继续使用 `checkpoint_ktcal_clean/stnet_best.pth`。

第三层空间核5x5→7x7可使参数量从2,085,420精确增加到2,282,028，接近论文2.281M；已支持旧5x5权重中心填充初始化。但7x7组val闭环明显退化，因此拒绝，不能仅凭参数量接近认定论文结构。实验JSON和配置保留在对应目录，消融权重已清理。

## 2026-07-30 权重清理

删除65个旧版、消融、中间epoch和重复参考权重，仅保留：

1. `checkpoint_ktcal_clean/stnet_best.pth`：离线batch最佳STNet。
2. `checkpoint_ktcal_clean/ncc_best.pth`：正式clean NCC。
3. `checkpoint_ktcal_stnet_laststep_aug/stnet_best.pth`：闭环最佳增强STNet。

三份权重均已重新加载验证。历史结果JSON、训练配置、rollout缓存和文档未删除。

## 2026-07-30 STNet调用预算优化（推荐低调用模式）

新增默认关闭的 `stnet_cooldown`、严重失败旁路、tier2开关和冷却回退策略。单纯调高阈值（conf=0.4、tau1/2=5/7、禁tier2）在val将调用降至37.3%，但最终误差恶化到4.84px；冷却期间使用hold或弱NCC也会级联漂移，均拒绝。

有效策略是保持原0.5/3/5阈值和tier2，在每次STNet调用后冷却4帧；冷却期间若本应触发，则回退到最近一次已确认的STNet模板中心并清零速度。仅当`confidence<0.2且边界峰`或`epsilon>=9px`时绕过冷却。32条val重复两次结果稳定：调用20.22%/20.16%，最终3.3401/3.3365px，基本等于无预算基线3.3365px。

独立test：

| 指标 | 高精度无预算 | 冷却4帧低调用 |
|------|-------------:|--------------:|
| 即时误差 | 2.7855px | **2.6910px** |
| 最终误差 | 2.7211px | **2.6829px** |
| 后处理误差 | 2.4908px | **2.4891px** |
| p5后处理 | **92.58%** | 91.95% |
| STNet调用 | 2348/4000（58.70%） | **918/4000（22.95%）** |
| tier3同步 | 54.95% | **22.53%** |
| hard reset | 1816 | **588** |
| CPU wall-clock | 16.79ms/帧 | **7.10ms/帧** |

调用量减少60.9%，CPU端到端耗时减少57.7%，平均误差没有损失且略有改善；代价是p5下降0.63个百分点。该模式最接近论文约20%触发率，推荐命令增加：`--stnet_cooldown 4 --severe_conf_threshold 0.2 --severe_epsilon_threshold 9 --cooldown_low_conf_mode anchor`。默认cooldown=0，仍复现原高精度调度。结果见 `checkpoint_ktcal_call_budget/test_cooldown4.json`。

## 2026-07-30 论文 P3/P5 与解码方式核查

论文 SEET 主表的 `p3=0.953、p5=0.996、MSE=1.30px` 对应完整3.248M TDTracker的100帧离线推理，并非当前KTCal低调用在线闭环。已从作者仓库恢复有效官方 `best_checkpoint.pth`，替换原损坏的22.1MB文件；恢复文件约11.6MB，可正常加载67个张量。

当前 `test_3et.h5` 实测：

| 模式 | 平均误差 | p3 | p5 |
|------|---------:|---:|---:|
| 官方TDTracker + 官方test有效3791帧 | 1.8047px | 85.91% | 99.00% |
| 官方TDTracker + 官方后处理 | 1.8030px | 85.97% | 99.00% |
| clean STNet全序列soft | **1.6997px** | **86.05%** | **98.83%** |
| clean STNet全序列argmax | 1.7783px | 85.25% | 98.65% |

低调用闭环固定32条val：soft后处理为`3.1834px / p3=74.31% / p5=85.66% / 调用20.22%`，argmax为`3.7387px / p3=70.03% / p5=84.84% / 调用20.97%`。argmax明确退化，正式配置维持soft。`test_ktcal.py`和`select_ncc_rollout.py`新增`--stnet_decode {soft,argmax}`，默认soft；验证结果见`checkpoint_ktcal_call_budget/val_{soft,argmax}_cooldown4*.json`。

结论：公开checkpoint和官方脚本在当前SEET文件上也不能直接达到论文主表，说明公开复现条件不完整。当前精度差距主要由完整TDTracker与轻量STNet、100帧双向离线上下文与16帧在线窗口之间的能力差造成，继续微调Kalman/NCC阈值无法把p3提升到95.3%。

## 2026-07-30 全序列教师蒸馏 + 3px精修（失败）

实现冻结clean STNet教师、温度KL和像素Smooth-L1精修，学生从正式增强STNet初始化。正式参数：窗口1/2/4/8/16、KL权重0.1/温度2、Huber权重0.05/beta3px、原归一化L2权重100、学习率1e-5。训练在epoch5 early stop，最佳epoch2。

固定val逐窗口结果：

| 窗口 | 原误差 / p3 / p5 | 蒸馏误差 / p3 / p5 |
|------|------------------:|--------------------:|
| W1 | **10.5270 / 18.75% / 30.47%** | 11.0823 / 10.16% / 25.78% |
| W2 | 5.4464 / 61.72% / **70.31%** | **5.3586 / 64.84%** / 69.53% |
| W4 | **5.3775 / 65.63% / 71.88%** | 5.4273 / 64.06% / 70.31% |
| W8 | **3.8962 / 75.78% / 84.38%** | 4.1135 / 74.22% / 82.03% |
| W16 | **3.2905 / 74.22% / 86.72%** | 3.3818 / 73.44% / 85.94% |

低调用闭环val：正式soft基线为`3.1834px / p3=74.31% / p5=85.66% / 调用20.22%`；蒸馏模型为`3.3471px / p3=70.81% / p5=84.38% / 调用20.28%`。除W2外全面退化，因此不运行test、不替换正式权重，失败权重已删除。根因是100帧双向教师在窗口终点利用未来帧，学生在线末帧无法获得同等信息，W1受损最明显。

代码保留通用蒸馏开关且默认关闭；`evaluate_stnet_windows.py`新增各窗口p3/p5。配置与结果保留在`checkpoint_ktcal_stnet_distill/`。

## 2026-07-30 单向 GRU STNet 复测（失败）

`models/STNet.py`新增`stnet_unidirectional`：单向GRU hidden=256、参数2.184M、FC头维度不变；`train/test_ktcal.py`、`select_ncc_rollout.py`、`evaluate_stnet_windows.py`同步接入`--stnet_unidirectional`。默认关闭。

从零训练（短窗1/2/4/8/16 + 末步监督 + coord_l2=100 + 增强，LR 2e-3、30 epoch、patience 6），第9轮最佳，第15轮早停。

| 指标 | 单向 | 双向增强基线 |
|------|------:|--------------:|
| 全序列每步 | 7.60px / p3=10.75% / p5=29.34% | 1.52px / p3=90.75% / p5=99.34% |
| 全序列末步 | 5.72px / p3=9.38% / p5=40.63% | 1.97px / p3=81.25% / p5=96.88% |
| W16 末步 | 7.14px / p3=10.94% / p5=41.41% | 3.29px / p3=74.22% / p5=86.72% |
| W1 末步 | 11.01px / p3=10.94% / p5=29.69% | 10.53px / p3=18.75% / p5=30.47% |

单向 GRU 全线明显退化，与用户之前的经验一致，未运行闭环val/test。失败权重已删除，配置与结果保留在 `checkpoint_ktcal_stnet_unidirectional/`。默认继续使用双向 Bi-GRU。
