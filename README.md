# STNet + KTCAL

事件相机注视点估计实验代码：**STNet** 负责序列坐标预测，**KTCAL** 通过模板匹配与卡尔曼滤波做跟踪与校正。本仓库公开的是我们的实现，不附带数据、训练权重或原基线模型。

## 环境与数据

建议 Python 3.9–3.11；在项目根目录安装：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-ktcal.txt
python -m unittest discover -s tests/ktcal_paper -t .
```

训练/验证示例使用 **3ET+2025 的已预处理数据**，由使用者自行获取并放在本机；下文假设 `data/3et/train_aug.h5`、`data/3et/test_aug.h5`。两者应分别含 `frames: [N,T,2,60,80]`、`label`（或 `labels`）: `[N,T,>=2]`，坐标为 80×60 图像尺度的归一化值。`test_200_200.h5` 是最终留出集，不用于调参或选择权重。数据和本地实验输出已加入 `.gitignore`。

## 3ET+2025：训练 STNet

从头训练（`--initialize-from` 可选，只接受**本项目的 STNet 检查点**；从头训练不保证达到已有权重的精度）：

```bash
python -m scripts.train_stnet_3et \
  --train-h5 data/3et/train_aug.h5 --val-h5 data/3et/test_aug.h5 \
  --output local_run/stnet --window 100 --readout attention \
  --selection-metric offline_soft_ma11 --epochs 60 --device cuda
```

训练在验证集选模，输出 `local_run/stnet/best.pt`、`history.json`、`summary.json`。双向模型的 100 帧批量推理是**离线**协议；`offline_soft_ma11` 在验证集按 soft decode + 每段独立 MA11 的 P3 选择权重。其他读出方式/选择指标见 `python -m scripts.train_stnet_3et --help`。请勿将离线选模结果当作在线实时精度。

## KTCAL 模板匹配训练

先用我们的 STNet 冻结预测生成训练/验证定位中心。matcher 训练还需要一份本项目 KTCAL `train-seq`（或 `train`）命令生成的检查点，作为 `--source-checkpoint`；其 `model_state` 和 `matcher_state` 必须存在，训练仅更新 matcher。AIS2024 数据可用 `python -m ktcal_paper prepare --source <AIS原始目录> --output data/ais` 和 `python -m ktcal_paper train-seq --manifest data/ais/manifest.json --output local_run/ais_ktcal --device cuda` 生成 `local_run/ais_ktcal/bidirectional_best.pt`；先核对相应目录内检查点字段再传参。若已经有我们自己的 matcher `best.pt`，无需重训。

```bash
python -m scripts.finetune_3et_matcher \
  --train-h5 data/3et/train_aug.h5 --validation-h5 data/3et/test_aug.h5 \
  --stnet-checkpoint local_run/stnet/best.pt \
  --source-checkpoint local_run/ais_ktcal/bidirectional_best.pt \
  --output local_run/matcher --device cuda
```

输出 `local_run/matcher/best.pt`。不要把上面的 AIS 与 3ET 指标混作同一数据集实验。

## 验证 STNet + KTCAL

```bash
python -m scripts.evaluate_3et_combined_ktcal \
  --h5 data/3et/test_aug.h5 \
  --stnet-checkpoint local_run/stnet/best.pt \
  --matcher-checkpoint local_run/matcher/best.pt \
  --output local_run/combined_val.json --stnet-decode soft \
  --offline-filter ma11 --device cuda
```

输出 JSON 中分别列出 STNet、KTCAL 的逐帧指标及可选**离线时序滤波**；`p3` 是 3 像素命中比例，`mean_error_px` 是 80×60 尺度的平均欧氏误差。批量预计算全段 STNet 后的结果是**离线精度**，`latency_comparable=false`：不要把该报告中的 `ktcal_online` 字段名理解成实时速度，也不要用它推导在线省时比例。验证集上确定配置后，可使用 `python -m scripts.evaluate_ktcal_official_holdout --help` 了解一次性留出集评估入口，不要用留出集选模。

需要测**因果在线路径**时，使用独立的 `python -m scripts.evaluate_3et_causal_online --help` 和因果训练的 16 帧 STNet 权重；在线精度/延迟与上面的 100 帧离线实验不可直接拼成同一模型的指标。端到端离线耗时需要在相同硬件、批量、预热与计时边界下单独测量，以上验证命令不输出可比较的离线平均耗时。

模板匹配质量诊断（仅验证数据）：`python -m scripts.diagnose_3et_template_matching --help`。AIS2024 的完整 KTCAL 在线实验入口：`python -m ktcal_paper --help`。所有结果应同时记录数据划分、权重哈希、解码/滤波配置和延迟测量协议。

## 仓库结构

- `models/STNet.py`：我们的序列 STNet；`ktcal_paper/stnet_sequence.py`：3ET/AIS 训练。
- `ktcal_paper/model.py`、`matching.py`、`tracker.py`：KTCAL 网络、模板匹配及门控/卡尔曼跟踪。
- `ktcal_paper/combined_3et.py`、`official_holdout.py`：验证与留出集评估。
- `scripts/`：训练、评估及诊断入口；`tests/ktcal_paper/`：单元/合成数据测试。
- `hardware/rv1106/`：论文 RV1106 CPU 快路径＋可选 RKNN NPU 接口、导出脚本及构建说明；未附 SDK、板端模型或硬件指标证明。

权重和预处理数据不随代码分发，克隆后无法直接得到本文已有实验数值；本仓库不声称 P3 已达到 91%。
