# RV1106 C++ 双路径实验实现

论文中的“硬件”指 RV1106 SoC：Cortex-A7 CPU 跑恒速 Kalman + 局部模板匹配，INT8 NPU 按门控运行 STNet。论文把 RTL/ASIC 列为未来工作，**不是** Verilog/FPGA 实现。本文代码仅提供可移植的板端 C++ 推理主循环和 RKNN Runtime 接口；不包含 Rockchip SDK、NPU 模型、训练权重或任何开发板实测数据。

## 输入和模型契约

- 每帧 `2×60×80` 单精度 float，按 CHW 连续排布，顺序为历史 16 帧、最早帧在前；第一帧不足 16 帧时重复首帧。所有坐标为 80×60 像素坐标。
- 从**本项目** matcher 检查点导出 float 权重（3×3 depthwise + 1×1 pointwise + ReLU），CPU 进行 30×25 模板 / 35×30 搜索窗的 cosine NCC，取硬峰值。仅提供论文 fine capture；不含 Python 的 coarse/亚像素/PSR 策略，因而两端精度不能直接等同。
- RKNN Runtime 输入需为 **单个 NCHW** `[1,16,2,60,80]`，输出 **两个**以此顺序排列的 float 可解码 logits `x[80],y[60]`；初始化即校验形状和输入布局，不符合则拒绝启动。`INPUT_DIVISOR` 是运行时除数；若使用下文的 ONNX 包装器，归一化已封装在模型里，运行时应设为 **1.0**，避免重复除。
- `export_stnet.py` 只接受训练配置标记 `causal_last_step=true, window=16` 的**本项目** STNet 检查点，输出取序列最后一帧的两个 SimDR logits，保留原模型的双向 GRU 权重（其输入严格只含当前和历史 16 帧），**不是**把双向 GRU 改成单向。仓库默认 100 帧双向离线 `best.pt` **不能**直接放到板端，更不能删掉反向 GRU 就当成同一模型。ONNX 仍需转换成 `.rknn`，Conv3D、GRU、五维输入的转换与量化是否受 RV1106 SDK 支持尚未验证；**转换失败时不能使用此板端后端**。

## 本机验证

在仓库根目录：

```sh
make -C hardware/rv1106 test
python -m unittest tests.ktcal_paper.test_hardware_export
```

`test_fast_path.cc` 测边界门控、Kalman、真实 CPU 特征匹配、16 帧缓存和异步结果丢弃；`test_rknn_backend.cc` 可用**官方 RKNN 头文件＋模拟 SDK 函数**额外做接口编译/输入输出检查，但其通过不代表设备可用。

## 板端构建与运行（需要您提供 SDK/模型/数据）

```sh
python -m hardware.rv1106.export_matcher \
  --checkpoint /path/to/matcher/best.pt --output /path/to/matcher.bin
python -m hardware.rv1106.export_stnet \
  --checkpoint /path/to/causal16/best.pt --output /path/to/causal16.onnx

# 使用适配 RV1106 的官方 RKNN-Toolkit2 将 ONNX 转换/校准为 INT8 模型；
# 请先逐帧比较 PyTorch → ONNX → RKNN 的两个 logits 与硬峰值坐标。
# 本仓库没有转换配置、已转换模型及板端精度验收结果。

# 路径应指向与 RV1106 固件版本匹配的官方 RKNN Runtime SDK 目录；
# CXX 可替换为 RV1106 的交叉编译器。
make -C hardware/rv1106 board CXX=arm-linux-gnueabihf-g++ \
  RKNN_INC=/path/to/rknn/include RKNN_LIB=/path/to/rknn/lib

# 数据格式：连续 float32 小端 CHW 帧，来源应与模型训练的预处理保持一致。
./hardware/rv1106/ktcal_rv1106 matcher.bin causal16.rknn frames-f32-chw.bin 1.0 \
  > predictions.csv
```

程序**从文件尽快读取**（未按真实事件时间戳或 4.4 ms 帧间隔节拍），输出每帧像素坐标、路线及该帧 `tick_ms`；后台异步计算时间、文件读取/前处理和结束时等待末尾任务的时间**不在这个 tick 计时中**，不能将其当作端到端实时延迟或与论文在线触发率直接比较。接实际摄像头时需由采集侧提供每帧时间戳与节拍。首次定位走同步网络；低误差 `<3px` 从局部匹配更新 KF；中误差 `[3,5)px` 输出 KF 预测并异步提交网络；严重误差 `>=5px`、边界峰或置信度 `<0.5` 同步重置。中误差任务最多一件；下一帧仍未完成时改走同步锚点，较老/被锚点覆盖的结果丢弃（包括它们的 NPU 异常）。该策略不是 Python 路径的回放（retroactive replay）：只能修正后续状态，已输出坐标不会改变。结束时等待末尾任务结算统计，该等待不是任何一帧的 `tick_ms`。

**局限：** 尚无 RV1106 真机、INT8 `.rknn` 或板端电流测量，本工程不证明论文的 0.46 ms、0.27 W、42.7 MB、P3 指标；部署前需要检查 RKNN 转换、输出量化/归一化、预测一致性、实际超时与端到端基准。论文的 `ε` 是两个**估计值**之间的差，不是真值误差；门槛 3px 不保证最终误差小于 3px。
