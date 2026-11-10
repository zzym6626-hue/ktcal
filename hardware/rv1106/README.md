# RV1106 实验实现

板端 C++：CPU 运行 Kalman＋局部模板匹配，RKNN Runtime 按门控调用 STNet。仓库未提供 Rockchip SDK、转换好的 `.rknn` 模型或开发板实测结果。

### 模型与输入

- Matcher 可由仓库权重 `weights/3et/matcher_best.pt` 导出。
- STNet 板端示例需要自行提供 `window=16, causal_last_step=true` 的检查点。导出的 ONNX 输入为 `[1,16,2,60,80]`，输出为最后一帧的 `x[80]`、`y[60]` logits。
- 输入文件是连续的小端 `float32` 帧，每帧按 `2×60×80` CHW 排布。ONNX 导出器已封装训练时的输入归一化，因此示例中的运行时除数为 `1.0`。

### 验证与运行

```sh
make -C hardware/rv1106 test
python -m unittest tests.ktcal_paper.test_hardware_export

python -m hardware.rv1106.export_matcher \
  --checkpoint weights/3et/matcher_best.pt --output matcher.bin
python -m hardware.rv1106.export_stnet \
  --checkpoint /path/to/causal16/best.pt --output causal16.onnx

# 使用匹配开发板的 RKNN-Toolkit2 将 ONNX 转换为 causal16.rknn，并核对输出。
make -C hardware/rv1106 board CXX=arm-linux-gnueabihf-g++ \
  RKNN_INC=/path/to/rknn/include RKNN_LIB=/path/to/rknn/lib
./hardware/rv1106/ktcal_rv1106 matcher.bin causal16.rknn frames-f32-chw.bin 1.0 \
  > predictions.csv
```

板端使用硬峰值匹配，与 Python 评测路径不完全相同。`tick_ms` 不包含文件读取和后台异步计算；尚未验证 Conv3D/GRU 到 RV1106 的模型转换、板端精度、端到端延迟及功耗。
