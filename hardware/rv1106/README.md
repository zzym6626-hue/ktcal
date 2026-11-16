# RV1106

默认使用仓库的100帧权重。以下命令从仓库根目录执行；转换需要在Linux安装与开发板匹配的RKNN-Toolkit2，编译需要对应Runtime SDK和ARM工具链。示例使用RV1106的`librknnmrt`，工具链须匹配板端libc；其他SDK可用`RKNN_RUNTIME=rknnrt`覆盖库名。

```sh
make -C hardware/rv1106 test
python -m unittest tests.ktcal_paper.test_hardware_export tests.ktcal_paper.test_hardware_sequence

pip install onnx onnxruntime

python -m hardware.rv1106.export_matcher \
  --checkpoint weights/3et/matcher_best.pt --output matcher.bin
python -m hardware.rv1106.export_stnet \
  --checkpoint weights/3et/stnet_best.pt --output stnet100.onnx
python -m hardware.rv1106.verify_stnet \
  --onnx stnet100.onnx --output onnx_verified.json
python -m hardware.rv1106.convert_rknn \
  --onnx stnet100.onnx --verified-report onnx_verified.json \
  --target rv1106 --quantization-dataset /path/to/train_inputs.txt --output stnet100.rknn

# 导出完整100帧记录，小端float32 N,T,C,H,W原始计数，不额外归一化
python -m hardware.rv1106.records export \
  --h5 data/3et/test_aug.h5 --output records.bin
make -C hardware/rv1106 board CXX=/path/to/board-toolchain/bin/arm-linux-uclibcgnueabihf-g++ \
  RKNN_INC=/path/to/rknn/include RKNN_LIB=/path/to/rknn/lib

# 将可执行文件、模型及records.bin复制到板端后运行
./ktcal_rv1106 matcher.bin stnet100.rknn records.bin > predictions.csv 2> timing.log

# 将predictions.csv取回主机，输出P3、P5、P10及平均误差
python -m hardware.rv1106.records score \
  --h5 data/3et/test_aug.h5 --predictions predictions.csv --output board_metrics.json
```

处理参数与默认测试命令一致；小误差不融合STNet，中误差校正或重置，大误差重置，每条记录独立处理。100帧网络预测全部先计算，KF路径比例不是网络跳过比例。

`timing.log`包含10视图、解码、匹配、KF、MA7，按100帧均摊；不含采集等待、读盘、模型加载和CSV输出，未排除首轮。它不是单帧即时响应时间。

RV1106转换须量化，`train_inputs.txt`每行填写一个训练集原始100帧输入`.npy`的路径，禁止用测试集校准。2026-11-16使用RKNN-Toolkit2 2.3.2的整图量化预检在编译阶段失败，尚无可用`.rknn`，不能直接运行上述板端命令。完整9300帧ONNX+C++精度对照已通过；主机验证不代表板端精度或耗时。证据见`docs/experiments/hardware_preflight_20261116_v1/`。

分段实验的复现命令见[`hardware_temporal_split_20261116_v1`](../../docs/experiments/hardware_temporal_split_20261116_v1/README.md)：六个卷积小图和读出图已完成SDK转换及100帧主机模拟，但尚未接入板端程序。读出文件约142MB且包含CPU回退，未上传该大文件；不能直接用七个小图替代上述整图模型。**真板运行、精度和延迟均未验证。**

旧16帧实验保留：导出加`--mode causal16`，编译用`make board-causal16`，需要独立16帧检查点。
