# RV1106等价分段导出试验（2026-11-16）

这是隔离可行性试验，不是已完成的板端后端。发布代码、权重和门控未改。
不训练模型，不读取test_200_200.h5，不推送。

## 完整9300帧结果

同一test_aug.h5的93条100帧记录，当前发布权重、10视图、soft温度0.65、
逆变换后逐轴median、每记录MA7。匹配37×32/模板30×25、confidence0.28、
PSR2.55；KF Qpos0.05/Qvel0.2/R32；小创新<3仅KF输出，中[3,5)按距离1
校正/回滚，大或不可靠重置。低档global融合始终为0。

| 验证路径 | P3 | P5 | P10 | 平均误差px |
| --- | --- | --- | --- | --- |
| 原STNet PyCPU参考 | 90.3118% | 96.1935% | 99.2473% | 1.629334 |
| 空间ONNX＋原时序CPU，STNet | 90.3118% | 96.1935% | 99.2473% | 1.629334 |
| 原STNet＋KTCAL PyCPU参考 | 91.1075% | 96.4516% | 99.2473% | 1.614943 |
| 空间ONNX＋原时序CPU，KTCAL | 91.1075% | 96.4516% | 99.2473% | 1.614943 |
| 空间INT8 SDK模拟＋原时序CPU，STNet | 90.1613% | 95.8925% | 98.4839% | 1.670797 |
| 空间INT8 SDK模拟＋原时序CPU，KTCAL | 90.7849% | 96.3763% | 99.2688% | 1.637600 |

浮点拆分逐帧P3/P5/P10分类、门控完全相同。最终坐标最大差4.3869e-5px，
logits最大差6.6757e-5。同logits C++解码最大差1.5259e-5px。
INT8模拟KTCAL的P3下降0.3226个百分点，**尚未保住91%目标**。
SDK模拟不是实际NPU精度，不可把浮点91.11%写为量化板端结果。

浮点门控：5643低档、26中档（3校正/23回滚）、3631高档。
INT8模拟：5647低档（60.72%）、32中档（4校正/28回滚）、3621高档。
每条记录验证低档global±10000反事实不影响输出；100帧网络仍全部先计算，
KF比例不是网络跳过比例。

原发布CUDA参考KTCAL的P3同为91.1075%，均误差1.614474px、1.044ms/帧。
这是上次RTX3090复测，不是本次分段或板端延迟。CPU与CUDA门控/微小数值
差异仍保留，未声称跨平台完全相同。

## 转换实测

原三个空间Conv3D的时间核为1，等价转为逐帧Conv2D，复制权重、bias、BN
参数/统计；eval折叠BN后ONNX各只有Conv＋ReLU。原密集时间卷积、池化、
100帧双向GRU、attention、FC不变，暂用主机PyTorchCPU。

| 阶段 | 固定NCHW输入 | 输出 | ONNX字节 | RKNN字节 | 退出码 |
| --- | --- | --- | --- | --- | --- |
| spatial1 | 1×2×60×80 | 1×32×60×80 | 13068 | 14548 | 0 |
| spatial2 | 1×32×20×26 | 1×64×20×26 | 205455 | 59849 | 0 |
| spatial3 | 1×64×6×8 | 1×128×6×8 | 820114 | 214530 | 0 |

RKNN-Toolkit2 2.3.2，target rv1106，INT8；三次build/export/init模拟成功。
日志将各ConvRelu安排为NPU，无之前整图12470MiB展开及分配崩溃。
InputOperator仍显示CPU，不宣称零CPU/零传输。小图总RKNN约289KB，
**不包含时序/GRU等CPU权重和运行内存**。

校准只读取train_aug.h5前8条记录原frames：原/镜像，帧0/25/50/75/99，
每阶段80样本；后续阶段使用原浮点模型中间特征。无训练label或测试校准。
小样本是转换探测，不是充分量化校准。完整SDK模拟每阶段调用93000次；
保留同一build对象，仅load_rknn不能提供这里的主机模拟计算图。

## 复现试验

仓库根目录执行，安装匹配Linux的Torch/ONNX/ORT/RKNN工具；输出目录须
不存在。当前校准脚本在CUDA主机生成。SDK各阶段最好使用独立临时工作目录。

```sh
export PYTHONPATH=.:docs/experiments/hardware_split_20261116_v1
P=docs/experiments/hardware_split_20261116_v1
O=/path/to/new_split_experiment
python "$P/split_probe.py" export --output-dir "$O/onnx"
python "$P/split_probe.py" calibrate --train-h5 /path/to/train_aug.h5 \
  --output-dir "$O/calibration"
for stage in 1 2 3; do
  python "$P/split_probe.py" convert --export-json "$O/onnx/export.json" \
    --calibration-json "$O/calibration/provenance.json" --stage "$stage" \
    --output-dir "$O/rknn_stage$stage"
done
python "$P/evaluate_split_full.py" --h5 /path/to/test_aug.h5 \
  --export-json "$O/onnx/export.json" --output-dir "$O/floating"
python "$P/evaluate_int8_simulator.py" --h5 /path/to/test_aug.h5 \
  --export-json "$O/onnx/export.json" --calibration-json "$O/calibration/provenance.json" \
  --output-dir "$O/simulator"
python "$P/verify_results.py" --h5 /path/to/test_aug.h5 \
  --floating-dir "$O/floating" --int8-dir "$O/simulator" \
  --published-report docs/experiments/hardware_preflight_20261116_v1/published_reference.json \
  --output "$O/independent.json"
```

脚本不接入原rknn_sequence_backend，不能用三个文件替代整图运行原板端
程序。未来需独立实现分段调度/CPU尾部/布局转换。

## 证据和限制

full9300_v1/report.json、int8_sim9300_v1/report.json及完整CSV，
independent_verification_v1.json：完整覆盖、独立像素计分、逐帧浮点分类。
rknn_stage*_v1.log/.exit：真实SDK编译和模拟，不是fixture。
train_calibration_v1/provenance.json：源SHA、每样本/列表SHA与frames-only。
source_sha256.json、probe_sha256.json：执行源码和权重散列。
主机套件255项（248通过/7跳过），新增等价单测4项通过；C++make和
ASAN/UBSAN通过。SDK桩测试不是真板测试。

**尚无RV1106实测精度、延迟、内存或功耗。** 主机时间包含模拟/Python/
参考检查/IO，两个CPU验证并行，不是性能基准。
空间stage1全100帧float输出61.44MB；暂留CPU的密集时间卷积，10视图每
记录23.494656G MAC（算术统计，不是时间实测）。小图转换成功不等于板端快。
建议下一步先用更充分的train-only中间特征校准保住P3，再探索等价时间卷积
1×1子图、CPU轻量时序尾部和分块内存。生产后端修改和发布需另行确认。
