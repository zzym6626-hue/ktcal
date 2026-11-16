# 硬件代码发布核验（2026-11-16）

基于 GitHub main `98c1c0c` 整理；只加入100帧硬件接口、分段导出试验及证据。
根目录README、模型实现、发布权重、默认评测参数均未修改。原开发工作区保留。

## 运行

通用接口命令见[硬件README](../../../hardware/rv1106/README.md)。
分段实验命令见[时间小图试验](../hardware_temporal_split_20261116_v1/README.md)。

```sh
python -m unittest discover -s tests -t .
make -C hardware/rv1106 test
export PYTHONPATH=.:docs/experiments/hardware_split_20261116_v1:docs/experiments/hardware_temporal_split_20261116_v1
python -m unittest discover -s docs/experiments/hardware_split_20261116_v1 -p test_split_probe.py -v
python -m unittest discover -s docs/experiments/hardware_temporal_split_20261116_v1 -p test_temporal_probe.py -v
```

## 发布树核验

- 主机套件189项：188通过、1项CUDA测试因无GPU跳过；原始日志保留线程警告。
- 空间及时间等价测试各4项通过；C++测试通过。
- 实际发布checkpoint加载后，原网络与两种浮点分段的100帧合成输入对照通过。
  这是CPU数值等价检查，不是全数据精度或板端速度测量。
- 权重SHA与现有发布一致；未读取任何数据集，也没有重新训练或替换权重。

## 原始实验与兼容处理

`hardware_sequence100`、`hardware_preflight`、`hardware_split`、
`hardware_temporal_split`四个20261116_v1目录保留实验时报告、日志、SHA。
其中“未推送”等文字描述的是实验执行时状态，不是本次发布后的状态。
历史套件255项来自开发工作区，本次发布树的套件是上述189项，不混用结果。

两项分段拒绝测试原来调用未发布的depthwise模型构造参数，本次改为直接
构造真实grouped Conv3D非法输入，拒绝断言未放宽。原测试保存在`.py.original`，
原SHA清单保存在`archive_sha256_original.json`；当前清单已重新生成。
模型、导出映射和历史模拟CSV均未改动。兼容修复前后的测试日志均保留于本目录。

## 限制

六个卷积小图及完整读出RKNN图的转换/模拟结果是已归档的Linux SDK实验；
本次发布核验未重跑RKNN或ONNX Runtime（本机未安装这些依赖）。
142MB读出RKNN文件未上传，CPU回退及内存风险未解除。
`verify_smoke.py`重新核验需要完整远端模型产物和同一test_aug数据，
单靠本仓库归档不能重新执行七图完整产物SHA核验。
现有`ktcal_rv1106`仍只接收整图，不能直接传入分段模型。
**未连接真板，板端运行、精度、延迟、内存和功耗均未验证。**

原始CSV的CRLF行尾及SDK日志尾空白按原样留存，不做格式化，以免改变证据SHA。
源代码、Markdown及JSON的提交差异空白检查通过。
