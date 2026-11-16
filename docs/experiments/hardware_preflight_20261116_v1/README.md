# 完整精度对照与RV1106转换预检（2026-11-16）

使用当前发布权重，不训练、不更改门控或参数、不覆盖权重、不提交或推送。仅使用`test_aug.h5`验证；RKNN校准只读取`train_aug.h5`前8条记录的原始帧，不读取标签，不读取最终保留测试集。

## 发布程序RTX3090复测

| 方法 | P3 | P5 | P10 | 平均误差 | 每帧耗时 |
| --- | --- | --- | --- | --- | --- |
| STNet | 90.31% | 96.18% | 99.25% | 1.629179px | 0.360ms |
| STNet+KTCAL | 91.11% | 96.45% | 99.25% | 1.614474px | 1.044ms |

完整93×100帧，预热后3次复测；KTCAL三次为1.044/1.050/1.043ms每帧，取中位数。计时为完整100帧处理均摊，包含所有10视图网络、软解码、中值、匹配、KF及MA7，不包含数据读盘、模型加载、评分、CUDA Graph捕获及采集等待。不是开发板测量。

门控：低档5648帧（60.73%），中档27帧（0.29%，其中4次校正、23次回滚），高档/初始化3625帧（38.98%）。**低档STNet融合0次**。低档KF路径比例不是跳过网络比例，100帧预测均已计算。精度及档位计数完全复现旧发布参考。

证据：`current_release_cuda_v1.json`、`.log`、`.exit`与`published_reference.json`。

## PyTorch与ONNX+C++完整对照

| CPU方法 | P3 | P5 | P10 | 平均误差 |
| --- | --- | --- | --- | --- |
| PyTorch STNet | 90.31% | 96.19% | 99.25% | 1.629334435px |
| ONNX+C++ STNet | 90.31% | 96.19% | 99.25% | 1.629334260px |
| PyTorch STNet+KTCAL | 91.11% | 96.45% | 99.25% | 1.614943217px |
| ONNX+C++ STNet+KTCAL | 91.11% | 96.45% | 99.25% | 1.614943135px |

全部9300帧验证通过，各记录10视图输入逐字节相同，坐标误差在限值内，所有门控档位/计数相同；单独从CSV复算，每帧P3/P5/P10阈值分类全部一致。最大最终坐标差0.000110626px。相同ONNX logits的C++解码误差最多0.000015259px。

CPU低档5643、中档26（3次校正、23次回滚）、高档3631，低档融合0；每记录低档对应的STNet坐标改动±10000px后，原始输出及档位完全不变。

CPU与CUDA存在微小后端差异，不声称跨CPU/CUDA所有坐标或门控一致；以上阶段对照使用相同CPU参考，另用真实CUDA发布程序复核历史结果。CPU STNet P5比CUDA多一帧，KTCAL P3/P5/P10相同。

CPU单次统计：ONNX10视图推理均摊6.391ms/帧；生产C++视图变换、解码、匹配、KF、MA7加重放输入核对1.321ms/帧。这不是严谨延迟基准，不包含读盘/Python变换/进程启动且无预热，不可当作RV1106耗时。

证据：`full9300_v2/report.json`、`predictions.csv`、`full9300_independent_verification_v1.json`、`cpp_csv_scored_v3.json`。失败首轮与第41条记录浮点诊断均保留，见`full9300.log`、`record41_numerical_diagnosis_v1.json`与`REVIEW.md`。只调整验证容差，没有更改模型或生产推理。

## 实际RKNN预检：失败，未生成可用模型

AutoDL独立venv：Python3.8.20、Torch2.1.0、ONNX1.16.2、ONNX Runtime1.19.2、RKNN-Toolkit2 2.3.2，未修改原TDTracker环境。

1. 非量化转换：`build`拒绝`rv1106`的`do_quantization=False`，退出码1。
2. 量化转换：仅训练记录0..7及镜像，共16个原始100帧样本；构建将Conv3D展开为大通道Conv2D，报告中间权重12470MiB（原ONNX仅8379481字节），GRU/归一化/部分Gather/Transpose回落CPU。
3. 原生编译器报告`failed to malloc cpu memory, size:18446744072822027648`并SIGSEGV，退出139。没有`.rknn`或成功sidecar。日志不能证明精确SDK内部缺陷，但明确证明此次整图转换失败。

证据：`rknn_preflight.json`、`rknn_conversion_v1.log`、`rknn_conversion_quant_v2.log`、对应`.exit`、`rknn_error_excerpt.log`及`train_calibration_v1/provenance.json`。

**目前不能将整模型直接部署到RV1106，更没有板端精度、耗时、内存或功耗。** 下一步建议不改训练权重，先研究等价分段导出：空间Conv2D走NPU，时序/GRU在CPU或单独子图；必须重新核验完整流程与量化精度，不能仅缩短帧数或跳过算子校验。这是后续建议，本次未实现。

## 重现验证

`evaluate_full.py`和`full_bridge.cc`放入仓库下独立目录，Linux安装`onnx`、`onnxruntime`和g++，从仓库根目录运行：

```sh
python -m hardware.rv1106.export_stnet --checkpoint weights/3et/stnet_best.pt --output stnet100.onnx
python -m hardware.rv1106.verify_stnet --onnx stnet100.onnx --output onnx_verified.json
PYTHONPATH=. python docs/experiments/hardware_preflight_20261116_v1/evaluate_full.py \
  --h5 data/3et/test_aug.h5 --onnx stnet100.onnx --output-dir local_run/hardware_full_new
python -m hardware.rv1106.records score --h5 data/3et/test_aug.h5 \
  --predictions local_run/hardware_full_new/predictions.csv --output local_run/hardware_full_scored_new.json
python -m scripts.evaluate_3et --h5 data/3et/test_aug.h5 --output local_run/release_cuda_new.json
```

输出路径必须全新。`diagnose_record41.py`、`prepare_calibration.py`和`verify_full_csv.py`为此次隔离AutoDL目录的诊断脚本，固定实验路径，不是通用发布CLI。权重与验证代码散列见`source_sha256_v4.json`和`source_hash_verification.json`。

本机完整测试255项（248通过、7跳过），生产C++测试通过，权重散列未改变。仅文档更新实际转换限制，代码未经提交或推送。
