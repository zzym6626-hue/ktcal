# KTCal 论文软件实验

在 `IdeaProjects/tdtracker` 新增 `ktcal_paper`，按 `conference_latex_template (2).pdf` 明确给出的训练/推理规则实现。旧模型、脚本、权重和已有推理结果保留。

当前可实测的数据为 **3ET+ AIS 2024**，不是论文表 I–III 中的 SEET、3ET+2025、Ini-30。新的代码和本机结果不能直接证明那些表格已复现。论文未给出的实现决定与内部矛盾见 `docs/ktcal_paper/design.md`，代码链路见 `.joyspec/knowledge/ktcal-paper-codemap.md`。

## 已实现

- W=16、2×60×80 输入，空间/时间分离卷积，单层 BiGRU128，末步直接回归坐标。
- 训练时监督双向和前向支路；导出原训练好的前向 GRU 和坐标头，训练集上微调两轮。评估入口拒绝尚未完成微调的检查点。
- 3×3 深度可分离特征、矩形局部余弦匹配，训练 softargmax / 推理 hardargmax。
- 像素单位 Kalman、3/5像素门控、真实异步中误差校正和同步强制重置；保留源帧、丢弃过时结果、回放受影响状态。
- subject 级训练/验证/留出划分、原标签状态保留、因果历史窗口、断点续训、JSON/CSV/Markdown 实验结果。
- 架构消融 `full`、`no_temporal`、`no_gru`、`full_conv`；STNet-only 和门限消融。

## 本机运行

在项目根目录执行。本机 `.venv` 链接复用原项目的 Python 环境。其他机器可以新建环境后安装 `requirements-ktcal.txt`。

```bash
./.venv/bin/python -B -m ktcal_paper --help
./.venv/bin/python -B -m unittest discover -s tests/ktcal_paper -v
```

已处理的数据在 `data/ktcal_paper_ais2024/`。若从原始 AIS 包重新处理，使用一个尚不存在的输出目录：

```bash
./.venv/bin/python -B -m ktcal_paper prepare \
  --source /Users/zhangyimeng.39/Downloads/event-based-eye-tracking-ais2024 \
  --output data/ktcal_paper_ais2024_new
```

真标签只有原始 `train` 目录的 `label.txt`。`test` 的 `label_zeros.txt` 不会用于训练或准确率；没有真实标签的帧不能提供实验精度。

完整配置的训练命令如下；`--device auto` 在有 CUDA 时使用 CUDA，否则使用 CPU。100轮是实现选择，论文未说明完整轮数。CPU耗时需先用试验估算。

```bash
./.venv/bin/python -B -m ktcal_paper train \
  --manifest data/ktcal_paper_ais2024/manifest.json \
  --output local_run/ktcal_paper/full_run \
  --epochs 100 --patience 20 --batch-size 16 \
  --stride 1 --val-stride 1 --device auto
```

本机 pilot 使用8轮主训练+2轮微调、训练/验证stride16；每个输入窗口仍包含连续16帧。每轮有1,440个训练窗口、384个验证窗口。subject划分在切窗前固定；留出subject为1、13，验证subject为8、11，其余9个subject训练。所有实际指标均以保存的配置为准。

可以用 `--stop-after-epochs 1` 保存并退出，再恢复；恢复时可提供迁移后的同一份manifest（文件内容不变）。检查点恢复模型、NCC、优化器、阶段、轮数和随机状态，不能通过更改配置继续同一个实验。

```bash
./.venv/bin/python -B -m ktcal_paper train \
  --resume local_run/ktcal_paper/full_run/last.pt \
  --manifest data/ktcal_paper_ais2024/manifest.json
```

训练结束后使用 `forward_best.pt`。它记录实际完成两轮微调，以及按验证集选中的权重来自哪一轮。`last.pt` 主要用于恢复，不应替代最佳模型来生成正式表格。

```bash
./.venv/bin/python -B -m ktcal_paper evaluate \
  --checkpoint local_run/ktcal_paper/full_run/forward_best.pt \
  --manifest data/ktcal_paper_ais2024/manifest.json \
  --output local_run/ktcal_paper/full_val \
  --mode both --split val --device auto
```

默认逐帧运行、batch=1计时；各记录先重置状态，评分包含全部有效坐标，不裁剪预测到图像边界。最初不足16帧的窗口左侧重复首帧，不使用未来帧。可用 `--max-frames 100` 做前缀试验，但该结果会明确记为受限覆盖。

## 软件计时与结果

`report.json` 保存环境、检查点哈希、数据划分、指标和逐帧时序；`metrics.csv` / `table.md` 只来自真实预测，`*_predictions.csv` 可逐帧核对。

- `mean_error_px` 为80×60尺度下欧氏距离均值；`mse_px2` 为两个坐标维度的均方误差；p3/p5/p10按距离≤3/5/10统计。
- `processing_ms` 为开始处理至发出的服务耗时；`online_latency_ms` 为到达至发出的延迟。
- 默认 `as_fast_as_possible` 在主线程开始处理时提供下一帧；`--realtime` 按10ms间隔到达，会把排队积压计入在线延迟。两种调度不可混在同一比较中。
- `correction_completion_ms` 从提交到后台真正完成，包含队列等待；`correction_application_ms` 还包含结果收取和状态回放。`async_events` 另有从帧到达计时的字段。
- `frame_deadline_misses` 是在线输出错过下一个bin；`async_deadline_misses` 是后台校正完成错过下一个bin。
- `retrospective_unavailable_at_emission` 只能作诊断，不能冒充在线精度。`online_sync_resets` 与 `retrospective_resets` 分开统计。

本机CPU数据不等同于论文RTX3090 FP32延迟。预处理、模型加载、预热和文件I/O不在模型/追踪逐帧计时内。未虚构FLOPs、功耗或硬件部署结果。

## 消融实验

`scripts/ktcal_software_suite.sh` 在同一真实标签manifest上训练四种架构，再评估STNet-only、完整闭环及论文阈值消融；执行时间约为四次完整训练加逐帧评估。输出目录必须是新目录：

```bash
bash scripts/ktcal_software_suite.sh \
  data/ktcal_paper_ais2024/manifest.json \
  local_run/ktcal_paper/ablation_suite
```

可设置 `KTCAL_PYTHON`、`KTCAL_DEVICE`、`KTCAL_EPOCHS`、`KTCAL_STRIDE`；默认使用项目虚拟环境、auto设备、100轮、stride1。软件套件不会把未运行的消融填成论文数值。

要复现论文原表，仍需对应数据的完整真标签、训练/测试划分和预处理协议；当前适配器仅实现AIS2024原始数据。已有100帧分组HDF5若缺少record/subject身份，不能直接随机划分后宣称无泄漏复现。

## 先优化3ET：双向GRU推理

当前只使用AIS2024。训练/验证每16帧选一个监督点，但每个输入仍含连续16帧。首轮前向微调模型在384个验证窗口上的平均误差为7.142px；这是验证集诊断，不是测试结果。

保留双向GRU进行推理，并从已有双向权重开一个独立实验：

```bash
./.venv/bin/python -B -m ktcal_paper train \
  --manifest data/ktcal_paper_ais2024/manifest.json \
  --initialize-from local_run/ktcal_paper/pilot_full/bidirectional_best.pt \
  --output local_run/ktcal_paper/bidir_cosine_new \
  --inference-mode bidirectional --epochs 16 --patience 8 \
  --batch-size 16 --stride 16 --val-stride 16 \
  --lr .0002 --scheduler cosine --min-lr .00002 --device cpu
```

此模式使用双向坐标头在验证集选模，输出 `bidirectional_best.pt`；没有前向导出或前向微调。历史窗口在当前帧结束，所以窗口内部反向计算不需要未来输入。来源checkpoint、学习率、逐轮指标与数据指纹均保存在新实验目录。评估命令与前向模式一致，传入双向检查点即可。报告会明确标出推理模式。

`data/`、`local_run/`、`.venv/` 不提交Git。先在train/val上确认改善，锁定配置以后再手动执行 `--split holdout`；当前不要运行完整消融套件打开holdout。其他数据集留待3ET效果稳定后接入。

### 验证诊断

只读取训练或验证划分，输出逐记录、原始标签状态、双向/前向支路指标及逐帧预测。此入口允许诊断训练中间权重，不输出部署速度，也不能选择holdout。

```bash
./.venv/bin/python -B -m ktcal_paper diagnose \
  --checkpoint local_run/ktcal_paper/pilot_full/forward_best.pt \
  --manifest data/ktcal_paper_ais2024/manifest.json \
  --output local_run/ktcal_paper/pilot_val_diagnostic \
  --split val --stride 16
```

加 `--plot` 可保存事件图、真值和预测叠加图（额外需要matplotlib）。使用 `--stride 1` 可诊断该划分的所有有效帧。`evaluate` 才用于逐帧在线/闭环计时，两种报告不能混作同一测速实验。

结构试验可增加 `--readout final_hidden`，用两个方向遍历历史窗口后的最终状态预测当前坐标。默认 `--readout last` 维持原末步读出。切换读出时使用新的输出目录；实际结果会记录读出方式，不能把这两个版本混为论文原配置。
