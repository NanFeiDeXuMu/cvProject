# 研究开展方案：单目度量深度估计与视频一致性

> AIAA 3201 — Project 3（Spring 2027，2 人一组）
> 依据：`Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf`
> 网页版：<https://claude.ai/artifact/UPZRVPVag9bbp8ZKG2Cpwh>（私有页面，需在页面 Share 菜单中分享给队友）

---

## 0. 一句话目标

在 **Middlebury** 上用经典双目几何建立“深度 = f·B / 视差”的直觉，在 **NYU Depth V2** 上定量评测一个零样本度量深度模型（UniDepthV2-S），在 **davis_rollercoaster.mp4** 前 100 帧上定性对比逐帧深度（Depth Anything V2-S）与视频深度（Video Depth Anything-S）；然后**只选一个**轻量改进，与未改动的方法做受控对比，并做一个深度应用 demo。

---

## 1. 研究问题与假设

| 编号 | 研究问题 | 对应部分 | 评估方式 |
|---|---|---|---|
| RQ1 | 经典 SGM 立体匹配恢复的视差/度量深度有多准？在哪些区域失效？ | Part 1.1 | Middlebury 视差 MAE(px) + 有效预测比例 |
| RQ2 | 零样本单目度量模型在 NYU 上的绝对尺度误差有多大？误差主要来自**全局尺度偏差**、**逐图尺度波动**，还是**边界/形状误差**？ | Part 2.1 | NYU AbsRel / RMSE / δ1 |
| RQ3 | 视频模型相比逐帧模型能在多大程度上减少闪烁与边缘抖动？还残留什么伪影？ | Part 1.2 + 2.2 | 同帧并排定性对比 + 时间剖面图 |
| RQ4 | 一个轻量改进能否带来可测量的提升？该提升是否可泛化？ | Part 3 | 与未改动方法在同一协议下对比 + 消融 |

**Part 3 主假设（推荐方向 B）**
- H1：UniDepthV2-S 在 NYU 室内场景上存在**系统性的全局尺度偏差**（例如与焦距估计误差相关），用 NYU train 拟合的单一全局尺度可降低 test 上的 AbsRel。
- H0（备择解释）：若误差主要由**逐图尺度波动**而非全局偏差主导，则全局标定只能带来边际收益甚至变差——这同样是可报告的结论，需要用受控实验和失败案例解释原因。

---

## 2. 总体技术路线

```
            ┌──────────────── Part 1: Baseline ────────────────┐
Middlebury ─┤ StereoSGBM → 视差 → Z = f·B/(d+doffs)/1000 → 评测 │
 (15 trainQ)└──────────────────────────────────────────────────┘

            ┌──────────────── Part 2: SOTA 复现 ───────────────┐
NYU test   ─┤ UniDepthV2-S (RGB-only, 无尺度对齐) → 480×640      │
 (654 张)   │   → Eigen crop [45:471, 41:601], 0.001–10 m → 指标  │
            └──────────────────────────────────────────────────┘

            ┌──────── Part 1.2 / 2.2: 视频（仅定性） ──────────┐
davis_      │ DA-V2-S 逐帧 (relative)   ┐                      │
rollercoaster┤                           ├→ 全片段固定归一化 →  │
 前100帧     │ VDA-S 离线 (relative)     ┘   并排视频 + 时间剖面 │
            └──────────────────────────────────────────────────┘

            ┌──────── Part 3: 一个改进 + 一个应用 ─────────────┐
            │ 推荐：B 度量尺度标定（train 拟合 → 冻结 → test）  │
            │ 应用：深度感知背景虚化 / 点云                     │
            └──────────────────────────────────────────────────┘
```

---

## 3. 环境与工程准备

### 3.1 硬件与软件（已检查本机）
- GPU：**NVIDIA RTX 5070 Ti 16 GB**（Blackwell 架构，sm_120），驱动 610.88。
- 本机已有 miniconda3，系统 Python 为 3.9.13。

**关键风险：** Blackwell 显卡需要 **CUDA 12.8 及以上编译的 PyTorch（≥ 2.7）**。三个官方仓库的 `requirements.txt` 可能锁定了较老的 torch/xformers 版本，直接安装会出现 `no kernel image is available for execution on the device`。
**对策：**
1. 新建一个 conda 环境（Python 3.10/3.11），先装 `torch`/`torchvision` 的 cu128 版本；
2. 安装各仓库其余依赖时**去掉 torch / torchvision / xformers 的版本锁定**；
3. xformers 视为可选（DINOv2 层在没有 xformers 时会回退到标准注意力），若装不上就不用，并在报告中写明；
4. 第一步先用 1 张图 / 1 段短视频跑通三个模型，确认 GPU 能正常推理再推进。

### 3.2 仓库结构（建议）
```
cvProject/
├── README.md                # 依赖、数据准备、权重、训练/评测命令、代表性结果
├── environment.yml
├── third_party/             # git submodule：Depth-Anything-V2, Video-Depth-Anything, UniDepth
├── data/                    # .gitignore：middlebury/, nyu/, video/
├── checkpoints/             # .gitignore：DA-V2-S, VDA-S 权重
├── src/
│   ├── stereo/              # run_sgbm.py, eval_stereo.py
│   ├── nyu/                 # extract_nyu.py, run_unidepth.py, eval_metric.py
│   ├── video/               # extract_frames.py, run_dav2_frames.py, run_vda.py, make_side_by_side.py
│   ├── improve/             # calibrate_scale.py（或 flow_fusion.py）
│   ├── app/                 # depth_blur.py, pointcloud.py
│   └── utils/               # metrics.py, pfm_io.py, viz.py, profiler.py
├── results/                 # 各实验的 csv/json（表格数据源，可复现）
├── figures/                 # 报告用图
└── report/                  # CVPR LaTeX 模板
```

原则：**所有预测都保存浮点结果（`.npy`/`.npz`）**，可视化与评测都从浮点结果生成；所有表格都从 `results/*.csv` 自动生成，避免手抄数字。

---

## 4. 数据准备

| 数据 | 内容 | 处理要点 |
|---|---|---|
| Middlebury v3 | 四分之一分辨率输入 ZIP（~30 MB）+ 左视图 GT ZIP（~13 MB）；只用 15 对 trainQ、`calib.txt`、`disp0GT.pfm` | PFM 是自底向上存储，读取后需**上下翻转**；GT 中 `inf` 表示未知；使用 **trainQ 自带的 calib**（已对应 quarter 分辨率） |
| NYU Depth V2 | 标注 `.mat`（~2.8 GB）+ `splits.mat` | 用 h5py 读取：`f["images"][i].transpose(2,1,0)`、`f["depths"][i].T`；`trainNdxs`（795）用于开发/标定，`testNdxs`（654）只用于最终评测；**索引减 1**；建议一次性导出为 `.npy`/`.png` 以加速后续实验 |
| 视频 | `Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4` | 按原帧率取**前 100 帧**；先确认原始分辨率，记录到 README |

> 数据范围严格按要求：不额外录视频、不引入其他数据集。

---

## 5. Part 1：Baseline —— 几何与逐帧深度

### 5.1 经典双目：视差 → 度量深度
1. **立体匹配**：对 15 对 trainQ 运行 OpenCV `StereoSGBM`（主方法）与 `StereoBM`（对照）。
   - `numDisparities` 取 calib 中 `ndisp` 向上取整到 16 的倍数；`blockSize` 3–7；`P1 = 8·C·bs²`，`P2 = 32·C·bs²`；`mode = SGBM_3WAY` 或 `HH`；开启 uniqueness 与 speckle 过滤。
   - **OpenCV 输出是定点数，必须先除以 16**。
2. **无效区域规则**（需在报告中明确写出）：除以 16 后 `d ≤ 0`（OpenCV 用 `minDisparity−1` 标记无效）视为无效；可选：左右一致性检查（`|d_L − d_R(warp)| > 1 px` 视为无效）作为诊断消融。
3. **度量转换**：`Z[m] = f · baseline / (d + doffs) / 1000`（baseline 单位 mm），无效视差不转换。
4. **评测**：在 GT 有限且为正的像素上，报告
   - 平均绝对视差误差（px，仅在“预测有效 ∩ GT 有效”上计算）；
   - 有效预测比例 = 有效预测数 / GT 有效像素数；
   - 可选：bad-2.0、运行时间。
   - 逐场景表 + 平均值；BM vs SGBM、有/无左右一致性作为小消融。
5. **可视化**：左图 / 预测视差 / GT 视差 / 误差图 / 无效掩码；深度图用同一色标。
   - 天然失败案例：弱纹理（如 Shelves、Vintage）、光照/曝光变化（PianoL、MotorcycleE）、细结构（Jadeplant、Pipes）、遮挡边界。

### 5.2 单目逐帧基线（Relative Depth）
1. 用 **Depth Anything V2-Small** 对视频前 100 帧**逐帧独立**推理，调用 `model.infer_image(...)` 并保存浮点输出。
   - ⚠️ 不要直接用官方 `run.py` / `run_video.py` 的可视化结果——它们**对每帧单独归一化**，会掩盖闪烁，违反课程要求。
2. 输出一律标注为 **“relative depth”**，不换算米制，也不在 NYU 上用 GT 对齐。
3. **时序诊断**：与 Video Depth Anything 的结果对比，描述闪烁、边缘不稳定、运动伪影。

---

## 6. Part 2：SOTA 复现 —— 度量深度与时间一致深度

### 6.1 单图度量深度：UniDepthV2-S（推荐）
- 加载：`UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")`，按仓库 `infer(rgb)` 示例推理。
- **只输入 RGB**（不提供 GT 内参），**不做任何测试集尺度拟合**。
- 评测协议（必须严格一致）：
  1. 预测双线性插值到 **480 × 640**；
  2. 裁剪 `[45:471, 41:601]`；
  3. 有效掩码：GT ∈ [0.001, 10] m；预测裁剪到 [0.001, 10]；
  4. 逐图计算 AbsRel（主指标）、RMSE、δ1（max(p/g, g/p) < 1.25），再对 **654 张等权平均**。
- 效率：输入尺寸、每图推理时间、峰值显存、精度（FP32/FP16）、硬件型号。
- 额外记录（为 Part 3 分析服务）：UniDepth 同时预测的**相机内参**（fx, fy），与 NYU 标定内参（fx ≈ 518.9）比较。
- 定性：RGB / GT / 预测 / 误差图 / 边界放大对比。
- 报告措辞：“在本课程协议下的复现结果”，不声称精确复现论文数字。

### 6.2 视频深度：Video Depth Anything-S
- 命令起点：`--encoder vits --max_len 100 --max_res 640 --save_npz`，默认 FP16，离线模式。显存不足时降低 `--input_size` 并记录。
- **公平对比设置**：
  - DA-V2-S 与 VDA-S 使用**完全相同的 100 帧**、相同的输入分辨率上限（建议都按 max_res 640 处理），记录各自 input size；
  - **同一调色板**（例如 `Spectral_r` 或 `inferno`）；
  - 每个方法在**整个片段上冻结一次归一化**（例如该方法全部 100 帧深度的第 2 / 98 百分位），绝不逐帧归一化；
  - 色标标注 “relative depth；颜色不代表共享的米制尺度”。
- **可视化工具**：
  1. 并排视频 RGB | DA-V2-S | VDA-S（必交付物）；
  2. **时间剖面图（x–t slice）**：固定某一行像素，把 100 帧堆叠成图——闪烁会表现为横向条纹，非常直观；
  3. 关键帧放大（轨道细结构、车体边缘、天空区域）。
- 效率：两种方法的每帧时间与峰值显存。

---

## 7. Part 3：改进方向选择（只做一个）

### 7.1 方向对比

| 方向 | 定量评测 | 工作量 | 风险 | 适配度 |
|---|---|---|---|---|
| **B 度量尺度标定** | ✅ NYU 全量指标 | 低 | 可能提升很小 | ⭐⭐⭐ 直接回应“度量尺度”主题，最稳 |
| A 光流引导融合 | ❌ 只能定性 + 运行时间（可选数值时序误差） | 中 | 遮挡与相对深度仿射歧义处理复杂 | ⭐⭐ 契合“视频一致性”主题 |
| C 边缘感知细化 | ✅ NYU 指标 | 中 | 容易把纹理拷进深度 | ⭐⭐ |
| D 不确定性平滑 | ✅/❌ | 中高 | 不确定性估计质量难保证 | ⭐ |
| E 高效部署 | ✅ 速度/显存/精度权衡 | 中 | Windows 下 ONNX/量化工具链兼容问题 | ⭐⭐ |
| F 深度应用改进 | ❌ 主要定性 | 低 | 难以量化 | ⭐ |

**推荐：方向 B（主）**；若团队更想突出视频，**方向 A 为备选**（见 7.3）。

### 7.2 方向 B 详细设计：度量尺度标定

**基本流程**
1. 在 NYU **train 795 张**上运行 UniDepthV2-S，按与 test 完全相同的掩码/裁剪协议得到 (pred, GT) 对；
2. 在 train 上拟合尺度参数，**冻结**；
3. 在 test 654 张上用冻结参数校正后，按原协议评测一次。test GT 不参与任何拟合或选择。

**候选标定形式（在 train 上用 5 折交叉验证选择，不看 test）**
| 变体 | 形式 | 拟合方式 |
|---|---|---|
| B1 | `d' = s · d` | 全体像素最小二乘 |
| B2 | `d' = s · d` | 逐图 median(g/p) 再取中位数（鲁棒） |
| B3 | `d' = s · d` | 对数域均值：`s = exp(mean(log g − log p))`（与 AbsRel/δ1 更匹配） |
| B4 | `d' = s · d^γ` | 对数域线性回归（深度相关尺度，检验远近尺度偏差是否不同） |
| B5 | `d' = s · d + t` | 仿射（用于说明过拟合风险） |

**分析实验（提升“分析深度”的关键）**
1. **误差分解（仅诊断，单独标注，不进入主表比较）**：比较 raw → 全局标定 → 逐图 oracle 中值尺度 三者的 AbsRel，量化“全局偏差”“逐图尺度波动”“形状/边界误差”各占多少。由此回答：全局标定理论上最多能修掉多少误差。
2. **尺度比分布**：画 train 与 test 的逐图尺度比 `median(g/p)` 直方图——若分布中心偏离 1 且方差小，H1 成立；若方差大，解释 H0。
3. **焦距—尺度相关性**：散点图展示 UniDepth 预测 fx 相对误差 vs 逐图尺度误差，检验“尺度误差来自内参估计”的假设。
   - 可选诊断：输入 NYU 标定内参后的结果（明确标为“使用额外相机信息的诊断”，不作为本方法）。
4. **标定集规模消融**：用 10 / 50 / 200 / 795 张 train 图拟合，观察 test 指标与参数稳定性。
5. **按场景类型分解**：利用 NYU 标注中的场景类别（卧室、厨房、办公室等）分组报告 AbsRel 变化，判断提升是否普遍或仅在部分场景有效。
6. **统计显著性**：逐图 AbsRel 差值的 bootstrap 95% 置信区间 + Wilcoxon 配对检验。
7. **泛化讨论**：说明全局尺度是对“NYU 这台 Kinect + 室内分布”的标定，迁移到其他相机/场景（如视频片段）不一定成立——这正是课程要求解释的 “whether the improvement generalizes”。

**公平性**：同一测试集、同一掩码、同一输出尺寸 480×640、保持 UniDepth 原预处理；报告标定前后的推理时间（标定本身开销可忽略）。

### 7.3 备选方向 A 要点（若选择视频方向）
- 光流：`torchvision.models.optical_flow.raft_large`（或 Farnebäck 作为对照）。
- 关键难点：DA-V2 输出是**逐帧仿射不变的相对深度**，融合前需先在高置信像素上把上一帧 warp 结果对齐到当前帧（拟合 scale/shift），否则融合会引入偏移。
- 遮挡：前后向光流一致性检查剔除不可靠像素；置信权重 `w = exp(−‖fb_err‖² / σ²)`。
- 评测：时间剖面图、并排视频、运行时间与显存；可选数值指标：光流 warp 后相邻帧深度误差。对照组：DA-V2-S 逐帧、VDA-S。
- 改进方法使用 **未改动方法（DA-V2-S）的归一化参数**。

---

## 8. 应用 Demo

**主应用：深度感知背景虚化**
- 输入：视频帧（VDA-S 深度）或 NYU 图（UniDepth 度量深度）。
- 方法：选定对焦深度 → 按 `|d − d_focus|` 计算模糊半径 → 分层高斯模糊 + 软掩码合成，避免边缘光晕。
- 对比：“同一应用 + 不同深度来源”（逐帧 vs 视频；改进前 vs 改进后），展示边缘与时序稳定性的差异。

**可选应用：点云**
- 用 UniDepthV2 度量深度 + 预测内参（或 NYU 标定内参）反投影为彩色点云，用 Open3D 渲染多视角截图。可直观展示尺度误差（墙面/地面是否平直、房间尺寸是否合理）。

---

## 9. 评测与报告指标总表

| 实验 | 指标 | 表/图 |
|---|---|---|
| Middlebury 立体 | MAE(px)、有效比例、（bad-2.0）、运行时间 | 表 1（逐场景 + 平均），图：视差/误差/无效掩码 |
| NYU 度量深度 | AbsRel（主）、RMSE、δ1；输入尺寸、时间、峰值显存 | 表 2：UniDepthV2-S raw vs +改进 |
| Part 3 消融 | 同上 + 置信区间 | 表 3：B1–B5、标定集规模；图：尺度比直方图、焦距散点图 |
| 视频对比 | 定性：时间平滑、边缘稳定、运动伪影；效率 | 图：并排关键帧、时间剖面图；表 4：效率 |
| 失败案例 | 分类：反光/透明、弱纹理、细结构、远距离/天空、快速运动/遮挡 | 图：3 个以上来自给定数据的困难案例 |

**计时规范**：预热 ≥ 5 次；`torch.cuda.synchronize()` 后计时；`torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated()` 记录峰值显存；统一注明 GPU 型号、精度、输入尺寸、batch size = 1。

---

## 10. 时间计划与分工

以 8 周为基准，请按实际截止日期等比例压缩/展开。

| 周 | 成员 A | 成员 B | 里程碑 |
|---|---|---|---|
| W1 | 精读 [1][2][6][7][8]；搭 conda 环境 | 精读 [3][4][5][9][10]；拉取三个仓库 | 三个模型各跑通 1 个样例 |
| W2 | Middlebury 下载、PFM 读取、SGBM + 评测 | 抽取视频帧、DA-V2-S 逐帧推理、VDA-S 推理 | 表 1 初版；两种视频深度 `.npy` |
| W3 | NYU 导出、评测脚本、UniDepthV2-S test 评测 | 固定归一化可视化、并排视频、时间剖面图 | 表 2 baseline；并排视频初版 |
| W4 | 交叉检查评测代码（互审）；train 集推理 | 效率测量脚本；应用 demo（背景虚化） | Part 1 & 2 全部完成 ✅ |
| W5 | Part 3：B1–B5 实现 + 5 折 CV | Part 3：误差分解、直方图、焦距散点图 | 冻结最终标定方案 |
| W6 | test 最终评测、显著性检验、场景分组 | 失败案例整理、点云（可选）、报告插图/流程图 | 所有表格与图定稿 |
| W7 | 报告：Method + Experiments | 报告：Intro + Related Work + README | 报告初稿（6–8 页） |
| W8 | 通读修改、核对引用 | 打包 `depth_results.zip`、清理仓库 | 提交 ✅ |

**互审机制**：每个指标脚本由另一位成员独立复核（例如用手算的小样例验证 AbsRel/δ1 与 Eigen crop 边界），避免评测协议错误导致整组数字作废。

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| RTX 5070 Ti（sm_120）与旧版 torch 不兼容 | 使用 torch cu128 构建；去除各仓库的 torch/xformers 版本锁定 |
| NYU `.mat` 2.8 GB，h5py 随机读取慢 | 一次性导出为 `.npy`；评测时用内存映射 |
| OpenCV 视差未除以 16 / PFM 未翻转 | 单元测试：与 GT 视差统计量粗对比，画叠加图目检 |
| 视频逐帧归一化导致“看不出闪烁” | 统一用 `viz.py` 的全片段固定归一化函数，禁止调用官方可视化 |
| Part 3 无提升 | 按要求用受控实验 + 误差分解 + 失败案例解释原因，结论同样有效 |
| 显存不足（VDA） | 降低 `--input_size` 并在报告中注明；改进方法保持同一分辨率 |
| 在 test 上“偷看”调参 | 规定 test 只在最终冻结后运行一次；所有选择在 train 上用 CV 完成 |

---

## 12. 提交物清单

- [ ] **PDF 报告**：CVPR LaTeX 模板，6–8 页（不含参考文献）；摘要末尾附公开 GitHub 链接；Related Work 引用阅读清单中全部论文 [1]–[10]，并引用所用数据集（Middlebury、NYU [12]）、预训练模型与外部仓库。
  - 结构：Abstract & Intro / Related Work / Method（含流程图）/ Experiments（定量表、定性图、消融、分类失败案例）/ Conclusion（局限与未来工作）。
- [ ] **公开 GitHub 仓库**：README 包含依赖、数据准备、权重下载、训练/评测命令、代表性可视化结果。不要向 Canvas 提交原始代码文件。
- [ ] **`depth_results.zip`**：
  - davis_rollercoaster 前 100 帧 RGB / 深度并排视频（DA-V2-S vs VDA-S，固定归一化、同一调色板）；
  - 一个应用输出（背景虚化或点云，来自给定图像或片段）。
- [ ] （可选）arXiv 上传：模板切换为 CAMERA-READY，并在 Canvas 首页注明 arXiv ID。

---

## 参考文献（课程阅读清单）

[1] Hirschmüller, Semi-Global Matching, TPAMI 2008 · [2] Geiger et al., KITTI, CVPR 2012 · [3] Ranftl et al., DPT, ICCV 2021 · [4] Yang et al., Depth Anything, CVPR 2024 · [5] Yang et al., Depth Anything V2, NeurIPS 2024 · [6] Piccinelli et al., UniDepth, CVPR 2024 · [7] Bochkovskii et al., Depth Pro, ICLR 2025 · [8] Piccinelli et al., UniDepthV2, arXiv:2502.20110 · [9] Teed & Deng, RAFT, ECCV 2020 · [10] Chen et al., Video Depth Anything, CVPR 2025 · [11] Sturm et al., TUM RGB-D, IROS 2012 · [12] Silberman et al., NYU Depth V2, ECCV 2012
