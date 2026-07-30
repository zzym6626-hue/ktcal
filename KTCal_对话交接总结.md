# KTCal 复现项目 — 对话交接总结

> 本文档用于新对话继续接手。包含原始需求、已完成工作、关键发现、未解决问题、重要数据。

## 一、原始需求

在 TDTracker（CVPR 2025 事件相机眼动追踪第三名方案，代码在 `/Users/zhangyimeng/Downloads/TDTracker-main`）代码基础上，复现论文 **KTCal**（闭环预测-校正跟踪框架，含 STNet 轻量时空网络）。

- 论文与审稿回复见桌面：`22565_KTCal_Closed_Loop_Predic.pdf`、`7GHk 领域主席对提交内容 22565 的元审查.docx`
- 数据集：SEET（train 320 序列 / test 40 序列，每序列 100 帧，事件帧 `[2,60,80]`）
- 环境：macOS，**纯 CPU（无 GPU）**，PyTorch 2.1.0；数据在 `./data/seet/{train,test}_3et.h5`

## 二、KTCal 论文核心机制

- **STNet**：解耦时空卷积(ITD) + Bi-GRU(ETD) + FC 头（论文 2.28M 参数）
- **三层触发**（W1）：ε<3px 用卡尔曼；3≤ε<5px 异步 STNet；ε≥5px 强制 STNet
- **回溯校正**（W1/W3）：σ>τ3 时校正卡尔曼
- **联合训练**（W4）：L_total = L_ST + 0.3·L_match
- **并行/异步**（论文核心）：三路径并发，STNet 异步不阻塞主路径
- 论文指标（SEET）：MSE **1.46px**，平均延迟 **1.17ms**（GPU），STNet 触发率 **~20%**

## 三、已完成的代码

- `models/STNet.py` — STNet（解耦时空卷积+Bi-GRU），~2.08M；`forward()` 输出全帧，`last_step()` 只输出最后一帧
- `models/kalman_filter.py` — 匀速卡尔曼，`update(z, r_scale)` 支持可变观测噪声
- `models/ncc_matcher.py` — 轻量 2层16ch 特征提取 + **逐窗口归一化互相关**
- `ktcal_tracker.py` — 三层调度 + 置信度加权融合 + 并行计时；`window_size=16`
- `train_ktcal.py` — 联合训练 + **早停**（--patience/--min_delta）+ 跨帧 match_loss
- `test_ktcal.py` — 闭环评估 + Savitzky-Golay 后处理 + `--batch_mode` 批处理评估
- 权重：`./checkpoint_ktcal/stnet_best.pth`、`ncc_best.pth`
- 进度文档：`KTCal复现进度.md`

## 四、修复过的关键 Bug

1. CUDA 硬编码 → `map_location='cpu'` + `--use_cpu`
2. STNet 每帧对 growing-100 窗口重算 O(T²) → 有界窗口 `window_size=16`（Bi-GRU 不能真正增量）
3. **NCC 坐标映射漏了 half_template** → ε 恒 ~14px，已修正 `img = search_tl + cell + half_t`
4. 模板漂移 → 模板只由 STNet 结果刷新（tier1 不更新）
5. match_loss 同帧裁剪 → 改**跨帧训练**
6. 裁剪越界崩溃 → 中心 clamp 到 [0,1]
7. **NCC 归一化 bug（关键）**：整块归一化 → 峰值被压低。改为**逐窗口归一化真正 NCC**，conf 中位数 0.37→0.61，>0.5 比例 14%→~65%

## 五、核心发现

### 延迟两种口径
- **关键路径**（并行下用户等待 = 卡尔曼+NCC）：**1.49 ms/帧**
- **论文口径**（总处理时间÷总帧数）：**14.75 ms/帧**（差论文 1.17ms 约 12.6×）

### STNet 单次慢是"调用方式浪费"
`last_step` 对整窗算但只取 1 帧，耗时随窗口线性:
- window=1: 2.38ms / 4: 7.92ms / 8: 11.61ms / 16: 24.06ms / 100: 146ms
- STNet 批处理 forward 100 帧: 148ms 总 = **1.49ms/帧**（比 TDTracker 6ms 更快，因更轻）

### 批处理 vs 闭环（本条件下最终对比）

| 模式 | 平均像素误差 | 延迟(顺序执行) | p5 精度 |
|------|-------------|---------------|---------|
| **STNet 批处理**(`--batch_mode`) | **1.65 px** | **约1.5 ms/帧** | **99.1%** |
| KTCal 闭环 + 后处理 | 3.81 px | 14.92 ms/帧 | 76.7% |
| 论文 SEET | 1.46 px | 1.17 ms | — |

**在本条件（CPU/离线/触发率56%）下，批处理全面优于闭环**。KTCal 闭环价值只在【真实流式 + GPU + 低触发率】才体现。

### 训练与后处理
- 500 轮上限 + 早停(patience=20)，**epoch 69 停**，STNet MSE **1.72px**
- Savitzky-Golay 后处理对闭环有效(4.16→3.81, -8.5%)，**批处理下反而变差**(1.65→1.95,已平滑)

### 置信度加权融合（已采纳）
- NCC 观测 `r_ncc=max(0.1, 1-conf)`；STNet 观测 `r_stnet=0.1`
- 卡尔曼重置 1105→**0**，MSE 持平、更稳健

## 六、当前状态

已实施 `--batch_mode`。用法：
```bash
python3 test_ktcal.py --test_h5_path ./data/seet/test_3et.h5 \
  --stnet_ckpt ./checkpoint_ktcal/stnet_best.pth \
  --ncc_ckpt ./checkpoint_ktcal/ncc_best.pth --use_cpu --batch_mode
```
- 批处理: 平均像素误差 1.65px / 约1.5ms/帧
- 闭环(默认，不加 --batch_mode): 后处理平均像素误差 3.81px / 记账快路径 1.50ms、真实顺序 wall-clock 14.92ms

## 七、未解决 / 待办

- [ ] STNet MSE 1.65 vs 论文 1.46 差距（数据集小 320 + CPU；可试数据增强/更长训练但收益递减）
- [ ] 闭环若要继续：触发率 56%→20% 是关键（受事件稀疏限制，NCC ε 中位数卡 3px 边界）
- [ ] 可选：扩大模板/搜索窗 11/21 → 15/31

## 八、协作注意（重要）

- 本轮对话多次出现**工具调用结果丢失**（编辑/命令声称成功但实际未落盘）。接手后**先用 grep/read 核实关键代码状态**再动手，尤其：`ncc_matcher.py` 归一化、`ktcal_tracker.py` 融合逻辑、`test_ktcal.py` 后处理块、`train_ktcal.py` 早停。
- CPU 训练：batch=8 稳定；batch=16 曾多次 OOM(exit 137)。
- 评估一次约 1 分钟（40 序列，闭环）；批处理约 10 秒。

## 九、2026-07-30 P0 实验可信度修复

- `train_ktcal.py` 不再读取 test 集做早停。默认以 seed=42 将 320 条 train 序列固定拆为 288 train / 32 val；`--test_h5_path` 仅兼容旧命令并明确忽略。
- 默认 CPU batch 改为 8；新增 `--seed`、`--val_ratio`、训练/验证 batch 限制和 `--no_save` 冒烟参数。
- 新 checkpoint 格式包含模型权重、epoch、最佳 val 平均像素误差、seed、完整配置和划分索引；`test_ktcal.py` 同时兼容旧纯 `state_dict`。
- 评估新增 `--output_json`，区分平均像素欧氏误差（px）与平均像素平方误差（px²），并报告真实顺序 wall-clock。
- 回归结果：离线 STNet 1.6467px；闭环 4.1615px，后处理 3.8055px；STNet 触发率 56.5%；真实顺序 wall-clock 14.92ms/帧。
- **重要：现有 `checkpoint_ktcal/*.pth` 是旧格式，且训练时曾使用 test 集选模，只可作为开发回归基线。正式结果必须使用新 288/32 划分从头训练后，再对 40 条 test 序列评估。**

## 十、2026-07-30 真实异步 KTCal

`ktcal_tracker.py` 已从“异步记账”改为真实并发：
- tier2 将带 `frame_id` 的 STNet 窗口快照提交给单后台 worker；worker 可将最多 4 个同长度窗口组成 batch。
- CPU 使用线程，CUDA 使用独立 stream；后台不修改 Kalman/模板，主线程轮询结果并按帧重放。
- tier3 和 NCC 低置信保持同步 STNet。这是论文三档语义，也避免全异步时结果晚到 36-38 帧导致 Kalman 发散。
- 支持 `r_stnet=0.1` 高置信融合和 `--hard_reset`；clean 权重消融已确认 `hard_reset --tau3 1` 的闭环精度更好，现作为正式闭环配置。
- 评估现在区分即时在线输出、最终闭环校正、在线关键路径、异步/同步 STNet 计算和真实 wall-clock。

旧开发权重在 SEET test 的回归结果：
- STNet batch：1.6467px，1.7032ms/帧。
- KTCal 即时在线：3.5309px；异步回收后：3.4755px；后处理：**3.2825px**。
- 在线关键路径：16.7720ms/帧；真实并发 wall-clock：**16.8318ms/帧**。
- tier2 异步仅 3.9%，计算量 1.4743ms/帧，平均晚到 4.90 帧/59.35ms。
- tier3 同步占 53.0%，计算量 14.5428ms/帧，是当前总延迟的主瓶颈。
- 全异步 tier3 的失败对照：即时误差 34-54px、最终约 15-16px，不可用。

clean 训练与独立 test 已完成，正式结果如下。降低延迟的下一优先级仍是降低 tier3/总 STNet 触发率（当前约 59%，论文约 20%）。

## 十一、2026-07-30 clean 权重最终结果

- 训练：seed=42，288 train / 32 val，epoch 70 早停；最佳 checkpoint 来自 epoch 50，val 平均像素误差 **1.4641px**。
- 正式权重：`checkpoint_ktcal_clean/stnet_best.pth`、`ncc_best.pth`；固定划分与配置见 `run_config.json`。
- 独立 test 离线 STNet：平均像素误差 **1.6997px**，p3=86.05%，p5=98.83%，CPU 1.65ms/帧；结果见 `test_batch.json`。
- 默认 soft fusion：即时在线 3.7494px，最终回放 3.7027px，后处理 **3.4416px**；触发率 58.7%，真实 wall-clock 18.83ms/帧；结果见 `test_closed.json`。
- 论文式 hard reset 消融：`tau3=1/1.5/3px` 的后处理误差分别为 **2.7375/2.7457/2.7679px**，重置次数为 1906/1654/1210。`tau3=1` 精度最好，但 47.7% 重置率远高于论文 14.8%。
- clean 权重下 tier3 仍主要来自 NCC 低置信（1181帧）和大误差（991帧），其中 NCC 边界峰 1639 帧。下一步不再继续扫 `tau3`，应处理 STNet 100帧训练/1~16帧在线推理不一致，并改进 NCC 边界峰抑制。

## 十二、2026-07-30 短窗口 STNet 闭环新基线

- `train_ktcal.py --stnet_only` 支持从 clean STNet 微调，训练时随机裁4/8/16帧；每个batch训练短窗，25% batch额外保留100帧损失，NCC完全冻结/跳过。
- 训练epoch10早停，保存epoch4。validation短窗误差从clean 3.3933px降到3.1553px（-7.0%），完整误差从1.4603px变为1.4969px。
- 新闭环STNet：`checkpoint_ktcal_stnet_short/stnet_best.pth`；继续搭配`checkpoint_ktcal_clean/ncc_best.pth`和`--hard_reset --tau3 1`。
- 独立test：即时2.8412px、最终2.7851px、后处理**2.5627px**、p5=91.40%；旧clean后处理同环境为2.7386px、p5=89.28%。
- 同环境wall-clock从25.7839降到25.1213ms/帧（-2.57%），STNet触发率从60.28%降到59.68%，reset从1897降到1846。
- batch离线误差由clean 1.6997px轻微退化到1.7145px。因此用途拆分：clean STNet用于batch离线最佳，短窗STNet用于KTCal闭环最佳，不覆盖彼此。
- NCC hard-negative与闭环选模微调均未超过clean NCC；正式闭环继续使用clean NCC。

## 十三、2026-07-30 真实 rollout NCC 实验

- 新增 `collect_ncc_rollout.py`、`train_ncc_rollout.py`、`select_ncc_rollout.py`，分别负责真实闭环状态采集、deployment-aware NCC训练和完整32条val闭环选模。
- rollout缓存：`checkpoint_ktcal_ncc_rollout/train_rollout.npz`（28,512帧）和 `val_rollout.npz`（3,168帧）。train/val均约 **24.2%** 帧GT不在11x11 NCC响应范围内；val大误差tier3中57.4%属于几何不可匹配。
- B组（Kalman center + GT template）val闭环从3.4418降至3.3743px，触发率52.13%降至51.31%；D组（历史template + failure-aware）val为3.3993px/51.25%。按val只选择B进入test。
- B组独立test：即时2.8424px、最终2.8063px、后处理2.6143px、触发58.50%、边界峰1574；正式clean NCC为2.8412/2.7851/2.5627px、触发59.68%、边界峰1673。
- 候选虽将边界峰降低5.9%、触发降低1.18个百分点，但后处理误差恶化0.0516px，未达到预设验收门槛，因此**不替换clean NCC**。结果见 `checkpoint_ktcal_ncc_rollout_gt/test_closed.json`。
- 该阶段正式配置为短窗STNet + clean NCC；后续已被最后一步增强STNet超过，见第十五节。
- 结论：固定11/21局部搜索存在约24%几何不可匹配上限。后续应改善预测中心或仅在失败时进行受控二阶段搜索，不再继续堆固定窗口NCC loss。

## 十四、2026-07-30 Kalman/调度/二阶段搜索实验

- GT轨迹理想常速度外推在val平均仅0.027px，几何越界主要来自状态污染而非运动过快。
- reset速度策略中val以preserve略优；独立test即时2.8245px、最终2.7890px、后处理2.5717px、触发58.10%，未超过正式最终/后处理精度，不采用。
- 低置信改为纯Kalman或hold会严重漂移；全部异步虽最终可回放到约3.42px，但在线误差巨大且触发升至75%~89%。正式低置信同步STNet必须保留。
- `velocity_decay=0`为val最佳阻尼；独立test即时2.8312px、最终2.7857px、后处理2.5835px、触发57.78%，仅降触发而损失后处理精度，不采用。Q/R小网格无额外因果收益。
- `analyze_ncc_fallback.py` 对31x31受控二阶段搜索离线分析：1539个候选的大窗整体误差13.91px；严格门控最多接受约1.4%总帧，且候选仍距Kalman约8.8px，因此不接入闭环。
- 以上实验参数均以可选项实现，默认仍保持旧行为。结果目录：`checkpoint_ktcal_kalman_reset_fix/`、`checkpoint_ktcal_low_conf_policy/`、`checkpoint_ktcal_kalman_tuning/`、`checkpoint_ktcal_ncc_fallback/`。
- 该阶段最优为短窗STNet + clean NCC，后处理2.5627px；随后已完成最后一步监督、L2/增强和2.28M结构实验，见第十五节。

## 十五、2026-07-30 STNet语义对齐与新正式最优

- `train_ktcal.py` 新增可选最后一步监督、坐标L2、窗口1/2/4/8/16，以及标签同步flip/rotation/shift/event-dropout；默认关闭，兼容旧训练。
- 最后一步KL在val主要改善W1/W2冷启动；加入100×归一化坐标L2后，W4/W8/W16和在线输出进一步小幅改善。
- 最佳增强配置：flip=0.5、rotation=±5°、shift=±2px、event dropout=3%，25% batch保留100帧最后步监督。val闭环在线/最终/后处理为3.4032/3.3369/3.1439px，均优于原短窗3.5072/3.4500/3.2515px。
- 独立test：即时**2.7855px**、最终**2.7211px**、后处理**2.4908px**、p5=**92.58%**、STNet触发率**58.70%**；原短窗为2.8412/2.7851/2.5627px、p5=91.40%、触发59.68%。
- 新闭环正式配置：`checkpoint_ktcal_stnet_laststep_aug/stnet_best.pth` + `checkpoint_ktcal_clean/ncc_best.pth` + `--hard_reset --tau3 1`。
- 离线batch仍使用 `checkpoint_ktcal_clean/stnet_best.pth`（1.6997px）；增强模型batch为1.7854px，只用于闭环。
- 第三层5x5改7x7可使参数量达到2,282,028，接近论文2.281M；但val闭环退化到3.5009px，已拒绝。可选结构与中心填充初始化代码保留，默认仍为5x5。
- 结果：`checkpoint_ktcal_stnet_laststep_aug/test_batch.json`、`test_closed.json`；其他消融的JSON和配置仍保留，但对应权重已按下述清理。

## 十六、2026-07-30 权重清理

已删除65个旧版、消融、中间epoch和重复参考 `.pth`，仅保留以下三份正式权重：

- `checkpoint_ktcal_clean/stnet_best.pth`：离线batch最佳STNet（1.6997px）。
- `checkpoint_ktcal_clean/ncc_best.pth`：正式clean NCC。
- `checkpoint_ktcal_stnet_laststep_aug/stnet_best.pth`：闭环最佳STNet（后处理2.4908px）。

历史章节提到的short、KL、L2、7x7、rollout NCC、Kalman参考权重均已删除；对应JSON、配置和实验结论保留，不应再将历史路径作为可运行checkpoint。

## 十七、2026-07-30 低调用预算模式

- 新增默认关闭的STNet冷却预算、严重失败旁路和锚点回退。单纯把`ncc_thresh/tau1/tau2`改成0.4/5/7会将val误差恶化到4.84px，不采用。
- 最佳策略保持原阈值0.5/3/5和tier2：STNet调用后冷却4帧；冷却触发时使用最近确认的STNet模板中心并清零速度；仅`confidence<0.2且边界峰`或`epsilon>=9px`绕过冷却。
- val重复结果：调用20.22%/20.16%，最终3.3401/3.3365px，与无预算3.3365px一致。
- 独立test：即时**2.6910px**、最终**2.6829px**、后处理**2.4891px**、p5=91.95%；STNet调用**918/4000=22.95%**，tier3=22.53%，reset=588，CPU wall-clock=7.10ms/帧。
- 无预算高精度模式为2.7855/2.7211/2.4908px、p5=92.58%、调用58.70%、wall-clock16.79ms/帧。低调用模式平均误差略优、p5低0.63个百分点，但调用减少60.9%、耗时减少57.7%，更接近论文约20%触发率。
- 推荐低调用参数：`--stnet_cooldown 4 --severe_conf_threshold 0.2 --severe_epsilon_threshold 9 --cooldown_low_conf_mode anchor`；默认cooldown=0仍保持旧行为。
- 结果：`checkpoint_ktcal_call_budget/test_cooldown4.json`。

## 十八、2026-07-30 论文 P3/P5 口径与官方 TDTracker 核查

- 用户给出的 SEET `p3=0.953、p5=0.996、MSE=1.30px` 来自 TDTracker 论文主表，属于完整3.248M TDTracker的100帧离线序列推理，不是当前约20%调用的KTCal在线闭环指标。
- 已从作者仓库恢复并验证根目录官方 `best_checkpoint.pth`（原本地22.1MB文件损坏；官方文件约11.6MB、67个张量），因此当前共有三份正式KTCal权重和一份官方TDTracker参考权重。
- 官方 `test.py` 在当前 `test_3et.h5` 上按其硬编码有效帧逻辑评估3791帧：`p3=0.8591、p5=0.9900、1.8047px`；官方后处理为`p3=0.8597、p5=0.9900、1.8030px`，未直接复现论文主表，表明公开checkpoint/脚本与论文SEET实验条件不完全一致。
- 同一clean STNet离线对照：soft expectation为`p3=0.8605、p5=0.98825、1.6997px`，argmax为`p3=0.8525、p5=0.9865、1.7783px`。
- 低调用闭环32条固定val：soft为后处理`3.1834px、p3=0.7431、p5=0.8566、调用20.22%`；argmax退化为`3.7387px、p3=0.7003、p5=0.8484、调用20.97%`。因此正式配置继续使用soft；新增`--stnet_decode`仅保留为可复现实验选项。
- 精度差距主要来自完整TDTracker与轻量STNet的模型能力、100帧双向离线推理与16帧在线窗口的上下文差异，不能靠继续调整Kalman/NCC阈值补齐。

## 十九、2026-07-30 全序列教师蒸馏实验（已拒绝）

- `train_ktcal.py`新增可选冻结教师、温度KL蒸馏和像素Smooth-L1精修：`--stnet_teacher_ckpt`、`--stnet_distill_weight`、`--stnet_distill_temperature`、`--stnet_pixel_huber_weight`、`--stnet_pixel_huber_beta`。默认权重均为0，不改变旧训练行为。
- 教师采用离线更好的`checkpoint_ktcal_clean/stnet_best.pth`，学生从正式增强模型初始化；训练窗口1/2/4/8/16，蒸馏权重0.1、温度2、Huber权重0.05/beta3px。训练第5轮early stop，最佳为epoch2。
- 固定val逐窗口仅W2略有改善；W1从10.5270px/p3=18.75%退化到11.0823px/10.16%，W16从3.2905px/74.22%退化到3.3818px/73.44%。
- 低调用闭环val从正式soft基线后处理`3.1834px、p3=74.31%、p5=85.66%、调用20.22%`退化到`3.3471px、p3=70.81%、p5=84.38%、调用20.28%`，因此未运行test、未替换正式模型。
- 失败原因：教师为100帧Bi-GRU双向输出，窗口终点包含学生在线推理不可见的未来信息；该特权信息未能稳定转移，尤其伤害W1。失败权重已删除，结果与配置保留在`checkpoint_ktcal_stnet_distill/`。
- `evaluate_stnet_windows.py`已补充全序列和各窗口p3/p5统计，便于后续直接按论文指标判断。

## 二十、2026-07-30 单向 GRU STNet（复测已拒绝）

- `models/STNet.py`增加`stnet_unidirectional`：单向GRU隐藏256维保持FC头兼容，参数量2,183,724；`train/test_ktcal.py`、`select_ncc_rollout.py`、`evaluate_stnet_windows.py`同步新增`--stnet_unidirectional`开关，默认关闭不影响旧行为。
- 复训recipe与`checkpoint_ktcal_stnet_laststep_aug`一致（短窗1/2/4/8/16 + 末步监督 + coord_l2=100 + 增强），只把GRU换单向；LR 2e-3从零训练30 epoch、patience 6。第9 epoch后无改进，第15 epoch早停。
- 最佳val：全序列每步 7.60px（p3=10.75%、p5=29.34%）、全序列末步 5.72px；短窗W1~W16 last-step在7.1~11.0px，p3始终≤11%、p5≤41%。与双向基线全序列每步1.52px/p5=99.34%、W16 3.29px/p5=86.72%差距巨大。
- 结论与前次实验一致：单向GRU在这个任务上明显退化。权重已删除，配置与结果保留在`checkpoint_ktcal_stnet_unidirectional/`。默认仍用双向Bi-GRU增强模型。
