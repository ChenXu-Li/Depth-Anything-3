# DA3-Streaming 完整使用指南

## 📖 目录

- [概述](#概述)
- [为什么使用 DA3-Streaming？](#为什么使用-da3-streaming)
- [快速开始](#快速开始)
- [完整工作流程](#完整工作流程)
- [脚本详解](#脚本详解)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [输出文件说明](#输出文件说明)
- [故障排除](#故障排除)
- [性能参考](#性能参考)

---

## 概述

DA3-Streaming 是 Depth Anything 3 的内存优化版本，通过分块处理（chunking）技术大幅降低显存使用，适合处理长视频序列和大规模场景。本项目提供了两个主要脚本：

1. **`infer_gs_streaming.py`**: 流式处理脚本，从图片文件夹生成深度、点云和相机姿态
2. **`streaming_to_gs.py`**: 后处理脚本，从流式处理结果生成3D高斯模型（PLY/视频）

---

## 为什么使用 DA3-Streaming？

### 适用场景

- ✅ **内存不足**: 遇到 "out of memory" 错误时
- ✅ **长序列处理**: 需要处理数千帧的长视频
- ✅ **大规模场景**: 处理高分辨率的大场景
- ✅ **稳定可靠**: 经过 KITTI、TUM RGB-D 等数据集验证

### 优势对比

| 特性 | 标准推理 (`infer_gs_from_images.py`) | DA3-Streaming (`infer_gs_streaming.py`) |
|:----:|:-----------------------------------:|:--------------------------------------:|
| **内存使用** | 高（一次性处理所有图片） | 低（分块处理，降低30-50%） |
| **适用场景** | 短序列、小场景 | 长序列、大场景 |
| **输出格式** | 直接生成高斯模型（PLY/视频） | 点云、相机姿态（需后处理生成高斯模型） |
| **处理速度** | 快（单次推理） | 中等（分块+对齐） |
| **精度** | 高 | 高（带回环优化） |

### 内存使用参考

根据 [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) 的测试结果：

| Chunk Size | 峰值显存 (KITTI 504×154) | 峰值显存 (TUM RGB-D 504×378) | 推荐 GPU |
|:----------:|:------------------------:|:----------------------------:|:--------:|
| 30         | ~11.5GB                  | ~18.7GB                      | 12GB+    |
| 60         | ~12.7GB                  | ~21.2GB                      | 16GB+    |
| 90         | ~14.3GB                  | ~25.1GB                      | 24GB+    |
| 120        | ~15.9GB                  | ~28.3GB                      | 32GB+    |

---

## 快速开始

### 1. 安装依赖

```bash
# 进入项目目录
cd /root/autodl-tmp/code/depth-anything-3

# 安装基础依赖
pip install -r requirements.txt

# 安装 DA3-Streaming 依赖
pip install -r da3_streaming/requirements.txt

# 安装 faiss (用于回环检测)
pip install faiss-cpu  # 或 faiss-gpu (如果有GPU)
```

### 2. 下载权重文件

```bash
# 下载 DA3-Streaming 所需的权重文件
bash da3_streaming/scripts/download_weights.sh
```

**注意**: 如果遇到网络问题（如 HTTP 503），可以：
- 使用 `--no-loop` 参数禁用回环检测（不需要 SALAD 权重）
- 手动下载权重文件（参考故障排除部分）

### 3. 运行流式处理

```bash
# 基本使用（推荐用于内存不足的情况）
python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 30 --no-loop
```

### 4. 生成高斯模型（可选）

```bash
# 从流式处理结果生成高斯模型
python streaming_to_gs.py exps/BridgeB300_2026-03-01-00-32-21 \
    --image-dir /root/autodl-tmp/data/BridgeB300 \
    --export-format gs_ply
```

---

## 完整工作流程

### 流程1: 仅流式处理（推荐用于内存优化）

```bash
# 步骤1: 运行流式处理
python infer_gs_streaming.py /path/to/images --chunk-size 30 --no-loop

# 输出: 相机姿态、点云、深度结果
# 位置: exps/<输出目录>/
```

**输出文件**:
- `camera_poses.txt`: 相机姿态
- `intrinsic.txt`: 相机内参
- `pcd/0_pcd.ply`: 点云文件
- `results_output/*.npz`: 深度和置信度结果（如果启用）

### 流程2: 流式处理 + 高斯模型生成（完整流程）

```bash
# 步骤1: 运行流式处理（保存深度结果）
python infer_gs_streaming.py /path/to/images \
    --chunk-size 30 \
    --no-loop \
    --save-depth-conf

# 步骤2: 从流式结果生成高斯模型
python streaming_to_gs.py exps/<输出目录> \
    --image-dir /path/to/images \
    --export-format gs_ply  # 或 gs_video
```

**最终输出**:
- 流式处理结果（点云、姿态等）
- 高斯模型文件（PLY 或视频）

---

## 脚本详解

### 1. `infer_gs_streaming.py` - 流式处理脚本

**功能**: 从图片文件夹进行流式深度推理，生成点云和相机姿态

**主要特点**:
- 分块处理，降低内存使用
- 支持回环检测（可选）
- 自动对齐分块结果
- 生成相机姿态和点云

**基本用法**:
```bash
python infer_gs_streaming.py <图片文件夹> [选项]
```

### 2. `streaming_to_gs.py` - 高斯模型生成脚本

**功能**: 从流式处理结果生成3D高斯模型（PLY点云或渲染视频）

**主要特点**:
- 使用流式处理得到的相机姿态
- 支持从NPZ文件或原始图片读取
- 支持PLY和视频两种导出格式
- 自动处理姿态转换

**基本用法**:
```bash
python streaming_to_gs.py <流式输出目录> [选项]
```

---

## 参数说明

### `infer_gs_streaming.py` 参数

#### 必需参数

- `image_dir`: 包含图片的文件夹路径

#### 内存优化参数

- `--chunk-size`: 分块大小
  - 选项: `30`, `60`, `90`, `120`
  - 默认: `60`
  - **建议**: 如果遇到内存不足，使用 `30`

- `--overlap`: 分块重叠大小
  - 默认: `chunk_size` 的一半
  - 说明: 重叠用于保证分块之间的连续性

#### 输出参数

- `--output-dir`: 输出目录路径
  - 默认: `./exps/<图片文件夹名>_<时间戳>`

- `--config`: 自定义配置文件路径
  - 默认: 使用 `base_config.yaml` 并自动调整参数

#### 处理选项

- `--loop-enable` / `--no-loop`: 启用/禁用回环检测
  - 默认: 启用
  - 说明: 回环检测可以提高长序列的精度，但需要下载额外权重

- `--save-depth-conf` / `--no-save-depth-conf`: 保存/不保存深度和置信度结果
  - 默认: 保存
  - 说明: 不保存可以节省磁盘空间

- `--delete-temp` / `--keep-temp`: 删除/保留临时文件
  - 默认: 删除
  - 说明: 临时文件可能占用大量空间（~50GB for 4500帧）

### `streaming_to_gs.py` 参数

#### 必需参数

- `streaming_output_dir`: DA3-Streaming 的输出目录路径

#### 输入参数

- `--image-dir`: 原始图片文件夹路径
  - 说明: 如果使用 `--use-streaming-poses`，可以从NPZ文件读取

- `--use-streaming-poses`: 使用流式处理得到的相机姿态和内参
  - 说明: 从NPZ文件读取图片和姿态信息

#### 模型参数

- `--model-dir`: 模型目录路径或Hugging Face仓库名
  - 默认: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`

- `--device`: 使用的设备
  - 选项: `cuda`, `cpu`
  - 默认: `cuda`

- `--process-res`: 处理分辨率
  - 默认: `504`

#### 导出参数

- `--export-dir`: 输出目录路径
  - 默认: `<streaming_output_dir>/gs_output`

- `--export-format`: 导出格式
  - 选项: `gs_ply`, `gs_video`
  - 默认: `gs_ply`

- `--gs-trj-mode`: [gs_video] 渲染轨迹模式
  - 选项: `smooth`, `linear`, `extend`
  - 默认: `smooth`

- `--gs-video-quality`: [gs_video] 视频质量
  - 选项: `low`, `medium`, `high`
  - 默认: `medium`

---

## 使用示例

### 示例1: 内存不足的情况（推荐）

```bash
# 使用最小chunk_size，禁用回环检测（避免网络问题）
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --chunk-size 30 \
    --no-loop
```

### 示例2: 平衡内存和精度

```bash
# 使用中等chunk_size，启用回环检测
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --chunk-size 60
```

### 示例3: 高性能GPU

```bash
# 使用大chunk_size以获得更好精度
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --chunk-size 120
```

### 示例4: 完整流程（流式处理 + 高斯模型）

```bash
# 步骤1: 流式处理（保存深度结果）
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --chunk-size 30 \
    --no-loop \
    --save-depth-conf

# 步骤2: 生成高斯模型PLY文件
python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 \
    --image-dir /root/autodl-tmp/data/road10 \
    --export-format gs_ply

# 或生成渲染视频
python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 \
    --image-dir /root/autodl-tmp/data/road10 \
    --export-format gs_video \
    --gs-trj-mode smooth \
    --gs-video-quality high
```

### 示例5: 使用NPZ文件（无需原始图片）

```bash
# 从NPZ文件读取图片和姿态
python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 \
    --use-streaming-poses \
    --export-format gs_ply
```

### 示例6: 节省磁盘空间

```bash
# 不保存深度结果，删除临时文件
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --chunk-size 30 \
    --no-loop \
    --no-save-depth-conf \
    --delete-temp
```

### 示例7: 自定义输出目录

```bash
# 流式处理
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --output-dir ./my_output/road10_streaming \
    --chunk-size 30 \
    --no-loop

# 生成高斯模型
python streaming_to_gs.py ./my_output/road10_streaming \
    --image-dir /root/autodl-tmp/data/road10 \
    --export-dir ./my_output/road10_gs \
    --export-format gs_ply
```

---

## 输出文件说明

### `infer_gs_streaming.py` 输出

运行完成后，会在输出目录生成以下文件：

#### 基本输出

- **`camera_poses.txt`**: 相机姿态文件
  - 格式: 每行包含一个4×4的C2W（相机到世界）矩阵的16个参数
  - 用途: 用于后续的高斯模型生成或3D重建

- **`intrinsic.txt`**: 相机内参文件
  - 格式: 每行包含 `fx fy cx cy`（焦距和主点坐标）
  - 用途: 用于后续的渲染和重建

- **`pcd/0_pcd.ply`**: 点云文件
  - 格式: PLY点云格式
  - 用途: 可以在 CloudCompare、MeshLab 等软件中查看

- **`pcd/combined_pcd.ply`**: 合并的点云文件（如果有多块）
  - 格式: PLY点云格式
  - 说明: 所有分块的点云合并结果

#### 额外输出（如果启用 `save_depth_conf_result`）

- **`results_output/*.npz`**: 每帧的RGB、深度、置信度和内参结果
  - 文件格式: NPZ (NumPy压缩格式)
  - 内容: `image`, `depth`, `conf`, `intrinsics`, `extrinsics` 等
  - 用途: 用于后续处理和可视化

#### 临时文件（默认删除）

- **`_tmp_results_unaligned/`**: 未对齐的分块结果
- **`_tmp_results_aligned/`**: 对齐后的分块结果
- **`_tmp_results_loop/`**: 回环检测的临时结果

### `streaming_to_gs.py` 输出

运行完成后，会在输出目录生成以下文件：

#### PLY格式 (`--export-format gs_ply`)

- **`gs_ply/0000.ply`**: 3D高斯模型PLY文件
  - 格式: PLY点云格式（包含高斯参数）
  - 用途: 可以在支持高斯模型的查看器中打开（如 SIBR Viewer）

#### 视频格式 (`--export-format gs_video`)

- **`gs_video/*.mp4`**: 渲染的视频文件
  - 格式: MP4视频
  - 内容: 从不同视角渲染的高斯模型
  - 轨迹模式: 根据 `--gs-trj-mode` 参数选择

---

## 故障排除

### 问题1: 仍然内存不足

**症状**: 即使使用 `chunk_size=30` 仍然报错 "out of memory"

**解决方案**:
1. 检查GPU显存是否被其他程序占用
   ```bash
   nvidia-smi
   ```
2. 降低处理分辨率（需要修改配置文件）
3. 使用更小的chunk_size（需要修改代码，不推荐）
4. 使用CPU模式（会很慢，不推荐）

### 问题2: 找不到 faiss 模块

**症状**: `ModuleNotFoundError: No module named 'faiss'`

**解决方案**:
```bash
# CPU版本（推荐）
pip install faiss-cpu

# GPU版本（如果有GPU）
pip install faiss-gpu
```

### 问题3: 找不到权重文件

**症状**: `FileNotFoundError: [Errno 2] No such file or directory: './weights/config.json'`

**解决方案**:
```bash
# 下载权重文件
bash da3_streaming/scripts/download_weights.sh

# 如果遇到网络问题，可以禁用回环检测
python infer_gs_streaming.py <图片目录> --no-loop
```

### 问题4: 下载权重时遇到 HTTP 503 错误

**症状**: `curl: (22) The requested URL returned error: 503 Service Unavailable`

**解决方案**:
1. **推荐方案**: 禁用回环检测（不需要SALAD权重）
   ```bash
   python infer_gs_streaming.py <图片目录> --no-loop
   ```

2. **替代方案**: 手动下载权重文件
   - 等待网络恢复后重试
   - 或使用代理下载

### 问题5: 磁盘空间不足

**症状**: `No space left on device`

**解决方案**:
1. 使用 `--no-save-depth-conf` 不保存深度结果
2. 使用 `--delete-temp` 自动删除临时文件
3. 确保有足够的磁盘空间（长序列可能需要50GB+）

### 问题6: 单chunk时点云文件未生成

**症状**: `pcd/` 目录为空

**解决方案**:
- 此问题已在最新版本修复
- 如果仍遇到，请确保使用最新代码

### 问题7: 生成高斯模型时找不到图片

**症状**: `❌ 错误: 图片文件夹不存在`

**解决方案**:
1. 使用 `--use-streaming-poses` 从NPZ文件读取
   ```bash
   python streaming_to_gs.py <输出目录> --use-streaming-poses
   ```

2. 或提供正确的图片目录路径
   ```bash
   python streaming_to_gs.py <输出目录> --image-dir <图片目录>
   ```

### 问题8: 姿态数量与图片数量不匹配

**症状**: `⚠️ 警告: 图片数量与姿态数量不匹配`

**解决方案**:
- 脚本会自动处理，使用最小数量
- 如果问题持续，检查流式处理是否完整运行

---

## 性能参考

### 处理速度

根据官方测试（KITTI序列，A100 GPU）：

| 方法 | 时间 (11,373帧) | FPS |
|:----:|:--------------:|:---:|
| VGGT-Long | 65min 08sec | 2.91 |
| Pi-Long | 60min 09sec | 3.15 |
| **DA3-Streaming** | **22min 17sec** | **8.51** |

### 内存使用

| Chunk Size | 峰值显存 (KITTI) | 峰值显存 (TUM RGB-D) |
|:----------:|:----------------:|:-------------------:|
| 30         | ~11.5GB          | ~18.7GB             |
| 60         | ~12.7GB          | ~21.2GB             |
| 90         | ~14.3GB          | ~25.1GB             |
| 120        | ~15.9GB          | ~28.3GB             |

### 磁盘空间

| 序列长度 | 临时文件 | 最终输出 |
|:--------:|:--------:|:--------:|
| 300帧    | ~5GB     | ~1GB     |
| 2700帧   | ~35GB    | ~5GB     |
| 4500帧   | ~50GB    | ~10GB    |

---

## 更多信息

- **DA3-Streaming README**: `da3_streaming/README.md`
- **项目主 README**: `README.md`
- **GitHub**: https://github.com/ByteDance-Seed/Depth-Anything-3
- **标准推理脚本说明**: `INFER_GS_README.md`
- **流式推理脚本说明**: `INFER_GS_STREAMING_README.md`

---

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间，长序列可能需要50GB+
2. **处理时间**: 分块处理需要更多时间，但内存使用大幅降低
3. **精度**: DA3-Streaming 通过回环检测和分块对齐保证精度
4. **输出格式**: DA3-Streaming 主要输出点云，需要后处理生成高斯模型
5. **网络问题**: 如果下载权重遇到问题，可以使用 `--no-loop` 禁用回环检测
6. **单chunk处理**: 当图片数量少于chunk_size时，会自动处理为单chunk，已优化处理逻辑

---

## 快速参考

### 常用命令

```bash
# 流式处理（最小内存）
python infer_gs_streaming.py <图片目录> --chunk-size 30 --no-loop

# 流式处理 + 高斯模型（完整流程）
python infer_gs_streaming.py <图片目录> --chunk-size 30 --no-loop --save-depth-conf
python streaming_to_gs.py exps/<输出目录> --image-dir <图片目录> --export-format gs_ply

# 从NPZ生成高斯模型（无需原始图片）
python streaming_to_gs.py exps/<输出目录> --use-streaming-poses --export-format gs_ply
```

### 文件结构

```
项目根目录/
├── infer_gs_streaming.py      # 流式处理脚本
├── streaming_to_gs.py          # 高斯模型生成脚本
├── da3_streaming/              # DA3-Streaming模块
│   ├── da3_streaming.py        # 核心处理逻辑
│   ├── configs/                # 配置文件
│   └── scripts/                # 工具脚本
└── exps/                       # 输出目录
    └── <输出目录>/
        ├── camera_poses.txt    # 相机姿态
        ├── intrinsic.txt       # 相机内参
        ├── pcd/                 # 点云文件
        ├── results_output/      # 深度结果（NPZ）
        └── gs_output/          # 高斯模型（如果生成）
```

---

**最后更新**: 2026-02-28
