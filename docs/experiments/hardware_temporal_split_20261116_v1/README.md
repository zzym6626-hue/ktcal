# 硬件优先：时间卷积小图探测（2026-11-16）

用户取消91%精度下限，目标改为硬件可运行性。本次是隔离试验，不改发布
权重、100帧、10视图和门控，不训练、不推送。低误差仍只输出KF。

## 已执行的结果

- 原空间卷积变成三个固定逐帧Conv2D；dense时间Conv3D(3,1,1)用
  [上一帧、当前帧、下一帧]通道拼接+1×1 Conv2D等价映射，记录两端补零。
  原BN/ReLU/池化保留。4项新等价测试通过，包括非平凡BN、批次边界、
  100帧/divisor一次、源tensor不变和不兼容结构拒绝。
- 六个ONNX导出并对照通过，三个新增时间小图真实rv1106 INT8编译和
  SDK主机模拟成功。ConvRelu/AvgPool在编译日志分配NPU。InputOperator
  仍CPU，不能称零传输。未再出现之前整图12470MiB展开崩溃。
- 完整100帧读出尾部单独导出ONNX，对照成功；也实际编译出
  readout100.rknn，并取得SDK模拟输出。**文件142,317,336字节**，SDK日志
  报告内部内存10000KB、权重52513.4KB；这些不是板端峰值RSS。
  GRU/exNorm/Gather等仍CPU回退，非全NPU，RV1106 mini runtime是否能实际
  执行这些CPU算子尚未验证。原SDK日志含Unknown op target警告/错误字样，
  尽管build/export/init/inference返回成功，不能隐去这些风险。
- 两条100帧烟雾链通过：六个INT8图+ORT浮点尾部；七个RKNN图（包括读出
  的CPU回退）+生产C++解码/中值/模板匹配/KF/门控/MA7。
  每个卷积图1000次、尾部10次，输出100个有限坐标。低档融合0，低档global
  ±10000反事实不改变输出，10视图raw输入逐字节核对。
- independent_smoke_v1.json独立核验两条链的按序100帧、P3/P5/P10/均误差、
  tier计数和模型SHA。**只测了test_aug第0条100帧，不是全9300帧最终精度。**
  日志里的约8–9秒是x86 SDK模拟+参考检查+IO，不是板端延迟。

| 新时间图 | 固定输入NCHW | 输出NCHW | RKNN字节 |
| --- | --- | --- | ---: |
| temporal1 | 1×96×60×80 | 1×32×20×26 | 15305 |
| temporal2 | 1×192×20×26 | 1×64×6×8 | 24758 |
| temporal3 | 1×384×6×8 | 1×128×6×8 | 64635 |

校准只读取train_aug的frames，前8条100帧原/镜像、帧0/25/50/75/99；每个
卷积子图80样本，读出16条特征。中间特征由原浮点checkpoint产生。没有
校准label/test读取。训练文件SHA和全部sample SHA在provenance中。

## 复现

复用上一轮split_probe.py所在目录；在具有匹配SDK的Linux环境，源码根目录
执行。输出目录须不存在。脚本为原执行版本，报告中的绝对路径应按实际
部署环境重建，不直接编辑SHA报告。

```sh
P=docs/experiments/hardware_temporal_split_20261116_v1
export PYTHONPATH=.:$P:docs/experiments/hardware_split_20261116_v1
python -B -m unittest discover -s "$P" -p test_temporal_probe.py -v
python -B "$P/temporal_probe.py" export --output-dir /path/to/new/export
python -B "$P/temporal_probe.py" calibrate \
  --train-h5 /path/to/train_aug.h5 --output-dir /path/to/new/calibration
# 真实转换需要RKNN-Toolkit2；所有六图可依次转换
python -B "$P/temporal_probe.py" convert --stage 2 \
  --export-json /path/to/new/export/export.json \
  --calibration-json /path/to/new/calibration/provenance.json \
  --output-dir /path/to/new/temporal1
```

probe_tail_rknn.py使用本次隔离远端路径，为可行性探测而非通用部署入口。
该读出导出产物及七图模拟产物留在AutoDL隔离目录，未将142MB模型放入仓库；
本归档保存SHA/日志/报告和六个小图，不构成完整板端模型包。

## 真板阻塞与下一步

SSH连接的是x86 AutoDL，未发现ARM交叉工具链、ADB或串口设备。本次未连接
RV1106，**没有板端精度、延迟、内存或功耗数据**，也未实现生产分段后端。
现有ktcal_rv1106入口要求一个整图模型，不能直接传七个小图替代。

真板测试需要开发板型号/内存、SSH或ADB访问方式，以及匹配的Runtime SDK
和工具链。优先验证readout100.rknn能否在mini runtime加载及CPU回退运行，
再接六图逐帧调度；若尾部模型/内存不可接受，改为六图NPU+轻量C++CPU尾部，
需另外实现并独立验证。不能因为服务器模拟成功就宣称板端已跑通。

## 验证与留存

255项现有测试：248通过、7跳过；C++make test通过。最初不带-t .的unittest
发现路径导致tests/ktcal_paper遮蔽源码包，45个import错误；加-t .后正确
完整通过。该失败日志保留，非删除/跳过失败测试。
转换v1因SDK工作目录切换后相对PYTHONPATH失效，未进入编译就退出1；
v2改绝对PYTHONPATH后退出0，原v1日志保留。ONNX固定batch GRU导出警告保留。

源码与发布权重SHA核对不变。REVIEW.md是作者自审，不是第三方板端证书。
archive_sha256.json列出本目录完整文件；不包含训练样本数组。
