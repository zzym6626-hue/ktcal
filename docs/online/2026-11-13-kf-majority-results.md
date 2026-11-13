# 当前网络关闭事件量门控后的KF占比对照

用户于2026-11-13调整方向：关注多数帧由匹配/KF跟踪的实际效果，不再要求达到论文指标。本轮不新增网络训练，只改变已有匹配置信度门控，完成四套配置各三轮完整因果在线复测。

## 完整结果

共用 `CurrentReadoutStreamingSTNet` 当前最佳权重、六视角（原图及水平±2px，各加水平翻转）、warm_start_steps=4、soft temperature=0.5、median融合、flip_axis_px=79.5。所有配置 `activity_reset_threshold=null`、无EMA、无PSR门控；q_pos=0.1、q_vel=0.5、r_px=32、reset_velocity_scale=1保持一致。

| 匹配置信度阈值 | 匹配/KF帧占比 | P3 | P5 | P10 | 平均误差/px | 平均ms/帧，三轮中位数 | 坐标头请求 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.28 | 81.9570% | 81.2043% | 90.6452% | 97.3441% | 2.415827 | 1.143265 | 1678/9300 |
| 0.45 | 70.5699% | 83.2366% | 91.5806% | 97.4086% | 2.290480 | 1.171044 | 2737/9300 |
| 0.60 | 55.8280% | 84.5269% | 92.3871% | 98.4731% | 2.087043 | 1.218979 | 4108/9300 |
| 0.82 | 19.7527% | 87.0968% | 94.1290% | 98.6022% | 1.890023 | 1.321508 | 7463/9300 |

“匹配/KF帧”对应small tier：PaperMatcher提供当前位置观测，KF更新位置/速度并输出，不读取当前STNet坐标头。medium需要网络校正；本次所有配置medium均为0。KF并非独立从图像定位。

每帧仍运行共享CNN/GRU以维护历史状态，所有配置均为9672次特征更新（9300真实输入+93×4首帧启动重复）。减少网络坐标校正主要省去坐标头及部分模板更新，不能将坐标头请求比例视为整个CNN的执行比例。

置信度0.60是这四套已测配置里，在KF帧占比超过50%的条件下P3最高的一套：5192帧匹配/KF，4108帧网络校正。置信度0.45则让KF帧占比达到约71%，代价是P3降至83.24%。这是有限配置对照，不宣称全局最优。

## 协议和验证

在 `test_aug.h5` 的93条独立record×100帧上逐帧处理，每次batch=1，每record重置状态。四套配置各测三轮，同配置三轮精度和路由完全一致。GPU计算进程快照均未观察到竞争，最终GPU检查为空。输入与权重float32，cuDNN允许TF32，matmul不允许TF32。

计时包含H2D、每帧CNN/GRU、请求的坐标头、解码、模板匹配、KF、D2H及GPU同步，包含首帧重复初始化；排除文件/权重加载、标签评分和CUDA Graph初次捕获。所有测试future_frames_used=0、labels_used_for_inference=false、global_predictions_precomputed=0，未读取最终留出集。

原87.6452%单轮版本包含事件量≥200强制网络重置。与相同空间设置但关闭事件门控的87.0968%对照相比，精度差0.5484个百分点。关闭该规则并不能使多数帧自然走KF；降低匹配置信度阈值才显著增加KF占比。两者为同一参与选型的验证集结果，不代表最终留出集泛化性能。

原始结果与配置：

- [置信度0.28三轮结果](kf_majority_20261113/current_kf_share_conf28_3x_20261113.json)、[原选型配置](kf_majority_20261113/current_kf_share_conf28_3x_20261113_config.json)
- [置信度0.45三轮结果](kf_majority_20261113/current_kf_share_conf45_3x_20261113.json)、[原选型配置](kf_majority_20261113/current_kf_share_conf45_3x_20261113_config.json)
- [置信度0.60三轮结果](kf_majority_20261113/current_kf_share_conf60_3x_20261113.json)、[原选型配置](kf_majority_20261113/current_kf_share_conf60_3x_20261113_config.json)
- [置信度0.82三轮结果](kf_majority_20261113/current_kf_share_conf82_3x_20261113.json)、[原选型配置](kf_majority_20261113/current_kf_share_conf82_3x_20261113_config.json)

四套结果的权重、匹配器、数据和源码SHA完全一致，且主要源码SHA与本地当前文件匹配；逐套核验了完整帧数、三轮路由/精度一致、无未来/标签推理、事件门控关闭及GPU竞争字段。保存的预测和标签形状均为93×100×2。

## 发布配置与权重

用户选定置信度0.60这档作为推荐在线配置。仓库提供 [3et_online_kf_majority.json](../../configs/3et_online_kf_majority.json) 和推理权重 `weights/3et/stnet_online_current_readout.pt`，只需另行准备数据即可运行。

推理文件约11MB，60个参数/缓冲张量与选中训练checkpoint逐元素一致，保留模型配置并移除优化器状态；见 [打包证据](kf_majority_20261113/weight_packaging.json)。原选型配置中的 `local_run/.../best.pt` 仅用于记录实验来源，复跑请使用仓库提供的发布配置。

发布权重重新完整运行三轮，精度与路由均与原选中模型一致：P3=84.5269%，平均误差2.087043px，匹配/KF占比55.82796%，坐标头请求4108/9300。三轮平均耗时分别为1.236612、1.224675、1.212439ms/帧，中位数1.224675ms/帧，未观察到其他GPU计算进程。见 [发布三轮复测](kf_majority_20261113/release_clean_3x.json)。打包前后9300帧预测最大差为0px，见 [完整预测一致性](kf_majority_20261113/prediction_consistency.json)。

发布前完整测试：本地297项通过，其中20项CUDA测试因无CUDA跳过；AutoDL上297项全部通过。日志见 [本地](kf_majority_20261113/local_tests.log) 和 [AutoDL](kf_majority_20261113/remote_tests.log)。

## 复跑

在仓库根目录运行，数据路径可替换，输出文件需使用新名称：

```bash
python -m scripts.evaluate_3et_online \
  --config configs/3et_online_kf_majority.json \
  --h5 /path/to/test_aug.h5 \
  --repeats 3 --save-predictions \
  --output local_run/current_kf_majority_recheck.json
```

训练来源为AutoDL的 `local_run/stream256_current_distill_aug_20261113/best.pt`，选中epoch13。发布权重的SHA与来源SHA记录在 [manifest](../../weights/3et/online_kf_majority_manifest.json)。
