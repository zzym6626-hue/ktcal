# 100帧硬件接口核验（2026-11-16）

本次仅适配当前发布权重的导出及C++处理接口，没有修改权重、训练模型或推送仓库。运行步骤见`hardware/rv1106/README.md`。

## 已执行的验证

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 完整主机测试 | 255项：248通过，7跳过，退出码0 | `host_tests.log` |
| C++旧接口及新SDK桩测试 | 通过，退出码0 | `cpp_tests.log` |
| 原PyTorch与ONNX Runtime，零输入及Poisson输入 | 最大logit差6.97e-5 | `onnx_synthetic_verified.json` |
| 原PyTorch与ONNX Runtime，真实100帧记录 | 最大logit差6.11e-5 | `onnx_real_record_verified.json` |
| 真实记录10视图、解码、C++匹配/KF/MA7与Python参考 | 最大坐标差2.76e-5px，门控档位一致，低档融合0次 | `real_record_pipeline_verified.json` |
| 使用官方完整SDK头编译backend与main | 主机编译通过，SDK调用通过测试桩；没有ARM链接或NPU执行 | `verification.json` |

真实输入仅为`test_aug.h5`第0条记录的帧，不含标签。该条记录有37次低档、0次中档、63次高档；中档校正与回滚另由确定性三档fixture验证，不能将这一条记录当作全数据集精度报告。未读取`test_200_200.h5`。

## 代码审查

审查限于本次100帧适配，不包含工作区已有实验改动。

| 路径与接口 | 检查结论 |
| --- | --- |
| `export_stnet.py::FullFrameLogits/StaticSpatialPool` | 权重不变，全帧输出，静态池化与原池化数值一致，divisor仅应用一次；旧16帧导出显式保留 |
| `verify_stnet.py::verify`、`convert_rknn.py::convert` | 校验模型/ONNX散列与固定形状；未核验或散列不匹配不得转换；转换失败不生成成功报告；finally释放SDK |
| `sequence_path.cc::predict_record` | 顺序执行10视图，逆镜像和位移后逐轴中值；与独立Python参考对照 |
| `SequenceGate::correct`、`SequenceTracker::track` | 低档只使用匹配更新KF，不融合、不刷新模板；中档校正或回滚，高档重置；记录间独立初始化，MA7不跨记录 |
| `rknn_sequence_backend.cc::run` | 检查输入/输出形状、有限值及SDK返回码；成功取得输出后复制并释放；拒绝16帧输入和截断记录 |
| `sequence_main.cc`、`records.py` | CSV包含全部记录和帧，评分拒绝重复/缺失/非有限坐标；计时明确包含后处理、不含采集等待、读盘及CSV输出 |

审查发现并修正两项问题，均先观察回归测试失败再修复：RV1106默认链接库改为`rknnmrt`并允许SDK覆盖；除`config`外，RKNN转换阶段必须明确返回0，不能将None当作成功。官方依据为Rockchip `rknn_model_zoo/3rdparty/CMakeLists.txt`的RV1106库选择，以及SDK头的接口声明。库、SDK头和原始输入没有加入仓库。

Python调用链辅助脚本的None提示经源码核对：配置缺失通过比较拒绝、`find_spec`显式判断None、argparse返回Namespace。脚本对内置函数的搜索不完整，因此关键调用链采用源码与测试手工核对，不声称完整图谱扫描。

## 未验证范围

- 实际转换因缺少`rknn` Python工具包停止，见`rknn_dependency_blocker.log`；没有生成可用`.rknn`。
- 尚无匹配板端SDK、ARM工具链及真实RV1106测试。Conv3D/GRU、5D输入、量化与运行时格式须实际核验。
- 这些结果只证明主机数值一致性，不能证明板端P3/P5/P10、耗时、内存或功耗，更不能把RTX3090耗时作为板端耗时。
- 权重散列和代码快照见`verification.json`，当前代码尚未提交或推送。
