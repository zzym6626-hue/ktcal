# KTCAL 在线优化结果：2026-11-13

基于重新拉取的 `main` 提交 `98c1c0cc61d23da3948a1b76b2bac572fe1df8f8`，本轮提高了因果在线 P3。默认两视角方案同时改善精度、平均误差和平均耗时；十视角方案进一步提高 P3，但耗时增加，部分大误差指标逊于两视角。

## 完整在线复测

数据为 `test_aug.h5` 的全部 93 个独立序列，每个 100 帧，共 9,300 帧。每次只处理一个序列，逐帧接收输入，初始化或门控需要时才调用 STNet。每套方案预热后完整运行三轮，三轮预测和路由一致；所有 GPU 进程快照均未观察到其他计算进程。

| 方案 | P3 | P5 | P10 | 平均误差 / px | 平均 ms/帧，三轮中位数 | 逐帧 P95 / ms，三轮中位数 | STNet 调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 此前最强在线基线 | 83.4194% | 91.8280% | 96.8065% | 2.249784 | 1.838222 | 2.136728 | 7,385 / 9,300 |
| 本轮默认：两视角、当前帧合成延拓 2 帧 | **83.9462%** | 92.0645% | **97.1613%** | **2.159363** | **1.790829** | **2.087868** | 7,325 / 9,300 |
| 本轮 P3 优先：十视角、不延拓 | **84.7312%** | **92.1183%** | 96.9462% | 2.168709 | 4.007524 | 4.899185 | 7,325 / 9,300 |

默认配置相对旧在线基线 P3 增加 **0.5269 个百分点**（多 49 帧达到 P3），平均误差下降约 **4.02%**，平均延迟下降约 **2.58%**。十视角增加 **1.3118 个百分点**（多 122 帧达到 P3），平均延迟为旧在线基线的约 **2.18 倍**。

这里“平均 ms/帧”先计算每轮全部序列的总处理时间 / 9,300，再取三轮中位数；它不同于逐帧 P50/P95。三轮完整平均耗时分别为：

| 方案 | 第 1 轮 / ms | 第 2 轮 / ms | 第 3 轮 / ms |
| --- | ---: | ---: | ---: |
| 旧在线基线 | 1.838179 | 1.849561 | 1.838222 |
| 两视角 | 1.802412 | 1.790829 | 1.778627 |
| 十视角 | 3.965996 | 4.009373 | 4.007524 |

环境：NVIDIA GeForce RTX 3090，PyTorch `2.1.0+cu121`，NumPy `1.24.1`，FP32，CPU threads=1。计时包含 H2D、历史缓存、实际请求的 STNet 视角、soft 解码、模板匹配、KF、D2H、因果 EMA，并在每帧结束同步 GPU。计时不包含磁盘/模型加载、标签评分、预热和 CUDA Graph 捕获，也不代表事件传感器累计时间、实际采集链路或板端性能。三轮复测用于检查计算稳定性，不是独立数据上的统计显著性验证。

原始证据：[旧在线基线](20261113/online_baseline_clean_3x_20261113.json)、[两视角](20261113/online_balanced_clean_3x_20261113.json)、[十视角](20261113/online_accuracy_clean_3x_20261113.json)。本轮之前带有其他 GPU context 的单轮实验只用于精度选型，不作为以上延迟证据。

## 这轮修改

1. **在线数据和训练对齐。** 历史窗口长度 16、采样间隔 2；左边不足时重复已收到的首帧。训练监督和验证只取当前帧，验证覆盖全部 9,300 帧。训练增加权重 `0.05` 的 soft 坐标 Huber 损失，选型改为当前帧 soft expectation 的 P3。单视角 STNet 的该指标从初始化的 80.1075% 升至 epoch 1 的 80.8172%；epoch 2 降至 80.0323%，因此采用 epoch 1。此处指标是单视角 STNet，不能与上表融合输出直接比较。
2. **迁移最新发布匹配器并保留真实按需调用。** 最终配置使用最新离线发布的 `matcher_best.pt`；小误差分支不调用当前 STNet。上表旧基线使用此前因果 STNet 和此前在线匹配器，因此整体提升不能全部归因于新损失。
3. **合成当前帧延拓与视角解码。** 两视角采用原图、水平翻转；复制已收到的当前帧 2 次作为双向 GRU 的边界条件，取原当前帧位置输出。十视角采用原图、水平翻转及 ±2 像素水平/垂直平移。延拓不读取未来、也不等待后续帧。输出 EMA 只在移动不超过 2.5 px 时以 alpha=0.5 平滑，结果不回写 KF 或模板。
4. **精简等价计算。** 当前帧 readout 只运行所需 GRU 方向和区间、只算当前坐标 head；使用 GPU 环形历史缓存、视角批处理、STNet/匹配器/模板 CUDA Graph。模板输出复制以避免缓存互相覆盖，捕获显式绑定设备与 stream；GPU 完成边界包含初始化和模板更新。attention/final_hidden 的读取语义保留。
5. **独立在线评测和可审阅配置。** 新入口拒绝覆盖既有报告，记录权重/数据/配置 SHA、实际网络调用、每帧延迟、GPU 干扰快照和在线协议字段。JSON 配置校验键、类型和 choices；命令行显式参数覆盖配置。

最终两套配置的 STNet 调用率均为 **78.76%**，实际处理视角分别为 14,650 和 73,250。放宽门控、缩短窗口的实验降低了精度，未作为默认；KF60 低阈值减少调用的好处在因果在线条件下不足以维持离线精度。

主要代码：[在线预测/解码](../../ktcal_paper/online_inference.py)、[在线入口](../../scripts/evaluate_3et_online.py)、[按需 tracker](../../ktcal_paper/offline_batch.py)、[训练](../../ktcal_paper/stnet_sequence.py)。`select_online_kf.py` 使用缓存坐标做选型，其运行时间不属于在线响应时间，候选必须由在线入口复测。

## 离线结果与论文差距

最新发布版本在同一完整验证集上的离线复测 P3 为 **91.1075%**，平均误差 **1.614474 px**，与发布精度一致。其原发布记录平均 **1.052424 ms/帧**；本轮复测为 **1.716326 ms/帧**，说明耗时受运行环境影响，不能混用两次记录。

该离线版本预先处理完整 100 帧，使用双向 GRU、attention、十视角和 centered MA7，当前输出可以依赖后续帧；按序列摊销的耗时不能作为逐帧在线响应时间。详见 [本轮离线复测](20261113/release_offline_recheck_20261113.json) 和 [原发布证据](../releases/kf60_20261112/README.md)。

本轮因果在线仍未达到此前对齐的论文目标约 92.1% P3、1.17 ms/帧，也未达到最新离线 P3。此报告不构成论文在线效果复现。两套方案均在 `test_aug.h5` 上选型，训练只读取 `train_aug.h5`；最终留出集 `test_200_200.h5` 未读取。继续泛化验证前应冻结配置，再使用独立留出数据；三轮重跑同一验证集不能替代这一步。

## 权重与复跑

两套配置共用 [在线权重](../../weights/3et/stnet_online_soft_pixel.pt)；原离线权重和默认离线入口保持可用。在线权重采用只保存推理参数的格式，57 个参数/缓冲张量与选中训练 checkpoint 逐元素一致；两套配置在全部 9,300 帧上的打包前后输出最大差均为 **0 px**。元数据见 [online_manifest.json](../../weights/3et/online_manifest.json)、[打包检查](20261113/online_weight_packaging_20261113.json)、[完整预测一致性及源码 SHA](20261113/online_final_prediction_consistency_20261113.json)。

| 文件 | SHA-256 |
| --- | --- |
| `weights/3et/stnet_online_soft_pixel.pt` | `20f5b9fc06ebff50203dd7216fcb70e8fcffb247e12ba8955573f1bf4a1df397` |
| 选中训练 checkpoint | `60c3053eead3e62e7fade70a8c3b73cce6583c52f56e626dd7b73927df415810` |
| `weights/3et/matcher_best.pt` | `a118055c2a5b64a33db0ff1155722a970655a696c7bfa2ee273c2313a91df77f` |
| `train_aug.h5` | `228b3bf77284587783c5ed1726e213f8799134edd48f665ade6546686b88b9a6` |
| `test_aug.h5` | `04ab5b392e250b79426bd6fc93b0f4464f000ed8ab38ac925ef8e500affffb7e` |

在仓库根目录运行默认配置：

```bash
python -m scripts.evaluate_3et_online \
  --config configs/3et_online.json \
  --h5 /path/to/test_aug.h5 \
  --repeats 3 --save-predictions \
  --output local_run/3et_online_recheck.json
```

P3 优先配置：

```bash
python -m scripts.evaluate_3et_online \
  --config configs/3et_online_accuracy.json \
  --h5 /path/to/test_aug.h5 \
  --repeats 3 --save-predictions \
  --output local_run/3et_online_accuracy_recheck.json
```

AutoDL 的本轮部署目录为 `/root/ktcal_online_20261113_v1`，Python 为 `/root/miniconda3/envs/TDTracker/bin/python`，数据为 `/root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/test_aug.h5`。需要在无其他 GPU 任务时复测；CPU eager 可用 `--device cpu`，不能用 CPU 耗时与此表比较。

训练选中权重的完整配置和每轮记录：[训练配置](20261113/soft_pixel_config.json)、[训练过程](20261113/soft_pixel_history.json)。在本轮 AutoDL 环境中可复跑：

```bash
python -m scripts.train_stnet_3et \
  --train-h5 /root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/train_aug.h5 \
  --val-h5 /root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/test_aug.h5 \
  --initialize-from /root/tdtracker_opt_20261112/local_run/stnet_ca16_stride2_dense8_epoch2_20261112.pt \
  --output local_run/causal_soft_pixel_retrain \
  --epochs 2 --patience 2 --window 16 --history-stride 2 \
  --causal-last-step --train-stride 16 --batch-size 32 \
  --lr 1e-5 --min-lr 5e-6 --readout last \
  --selection-metric causal_soft --pixel-loss-weight 0.05 \
  --preload-frames --seed 42 --threads 4 --device cuda
```

该训练命令依赖此前因果初始化 checkpoint，不能替换成发布的 attention 权重后仍称同一实验。直接使用打包的在线推理权重复测无需此前 checkpoint。

## 验证与审查

- 最新代码本地运行 **194 项测试，通过，5 项 CUDA 测试因本地无 CUDA 跳过**；AutoDL 运行 **194 项测试，全部通过，无跳过**。包含窗口环形缓存跨越、历史索引、未来扰动不影响前缀、调用门控、模板 alias、配置拒绝和离线兼容回归。日志：[本地](20261113/local_tests.log)、[AutoDL](20261113/remote_tests.log)。
- 发布、旧因果、选中训练三套真实权重 × 延拓 0/2，在变化历史输入下比较原始 eager 与优化 CUDA Graph。坐标最大差小于 `1e-4 px`，见 [GPU 等价性](20261113/online_equivalence_20261113.json)。独立代码审查另检验 45 组 STNet 当前 head，最大 logit 差 `2.98e-7`。
- 独立复审发现的 CUDA 设备/stream、异步计时和 JSON 布尔字符串问题均已修复并验证；最终代码复审没有必须修复的问题。文档与证据应一起审阅。

```bash
python -m unittest discover -s tests -t . -p 'test_*.py'
```

本地改动位于独立 worktree `ktcal-online-20261113`，分支 `optimize/online-from-kf60-20261113`。本轮代码、配置和权重已部署到上述 AutoDL 目录；未提交或推送 Git，原 `tdtracker` 工作区的已有修改未合入本轮分支。
