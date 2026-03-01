# DA3-Streaming 内存优化推理脚本使用说明

## 概述

`infer_gs_streaming.py` 是基于 **DA3-Streaming** 的一键运行脚本，专门用于处理内存不足的情况。它通过分块处理（chunking）来大幅降低显存使用，适合处理长视频序列和大规模场景。

## 为什么使用 DA3-Streaming？

当遇到 **"out of memory"** 错误时，DA3-Streaming 是理想的解决方案：

- ✅ **内存优化**: 通过分块处理，显存使用可降低 30-50%
- ✅ **长序列支持**: 可以处理数千帧的长视频
- ✅ **稳定可靠**: 经过 KITTI、TUM RGB-D 等数据集验证
- ✅ **速度快**: 在 A100 GPU 上达到 ~8.5 FPS

## 内存使用参考

根据 [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md) 的测试结果：

| Chunk Size | 峰值显存 (KITTI 504x154) | 峰值显存 (TUM RGB-D 504x378) | 推荐 GPU |
|:----------:|:------------------------:|:----------------------------:|:--------:|
| 30         | ~11.5GB                  | ~18.7GB                      | 12GB+    |
| 60         | ~12.7GB                  | ~21.2GB                      | 16GB+    |
| 90         | ~14.3GB                  | ~25.1GB                      | 24GB+    |
| 120        | ~15.9GB                  | ~28.3GB                      | 32GB+    |

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 DA3-Streaming 依赖
pip install -r da3_streaming/requirements.txt

# 安装 faiss (用于回环检测)
pip install faiss-cpu  # 或 faiss-gpu (如果有GPU)
```

### 2. 下载权重文件

```bash
bash da3_streaming/scripts/download_weights.sh
```

### 3. 运行脚本

#### 基本使用（推荐用于内存不足的情况）

```bash
# 使用最小chunk_size以节省内存
python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 30
```

#### 使用默认配置

```bash
python infer_gs_streaming.py /root/autodl-tmp/data/road10
```

## 完整参数说明

### 必需参数

- `image_dir`: 包含图片的文件夹路径

### 可选参数

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
  - 说明: 回环检测可以提高长序列的精度

- `--save-depth-conf` / `--no-save-depth-conf`: 保存/不保存深度和置信度结果
  - 默认: 保存
  - 说明: 不保存可以节省磁盘空间

- `--delete-temp` / `--keep-temp`: 删除/保留临时文件
  - 默认: 删除
  - 说明: 临时文件可能占用大量空间（~50GB for 4500帧）

## 使用示例

### 示例1: 内存不足的情况（推荐）

```bash
# 使用最小chunk_size
python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 30
```

### 示例2: 平衡内存和精度

```bash
# 使用中等chunk_size
python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 60
```

### 示例3: 高性能GPU

```bash
# 使用大chunk_size以获得更好精度
python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 120
```

### 示例4: 自定义输出目录

```bash
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --output-dir ./my_output/road10_streaming \
    --chunk-size 30
```

### 示例5: 节省磁盘空间

```bash
# 不保存深度结果，删除临时文件
python infer_gs_streaming.py /root/autodl-tmp/data/road10 \
    --chunk-size 30 \
    --no-save-depth-conf \
    --delete-temp
```

## 输出文件说明

运行完成后，会在输出目录生成以下文件：

### 基本输出

- `camera_poses.txt`: 相机姿态文件
  - 每行包含一个帧的外参矩阵参数

- `intrinsic.txt`: 相机内参文件
  - 每行包含一个帧的 fx, fy, cx, cy

- `pcd/combined_pcd.ply`: 合并的点云文件
  - 包含所有帧的3D点

### 额外输出（如果启用 `save_depth_conf_result`）

- `results_output/`: 包含每帧的RGB、深度、置信度和内参结果
  - 文件格式: NPZ

可以使用以下命令处理深度结果：

```bash
python da3_streaming/npz_output_process.py \
    --npz_folder ${OUTPUT_DIR}/results_output \
    --pose_file ${OUTPUT_DIR}/camera_poses.txt \
    --output_file ${OUTPUT_DIR}/output.ply
```

## 与标准推理脚本的对比

| 特性 | `infer_gs_from_images.py` | `infer_gs_streaming.py` |
|:----:|:-------------------------:|:----------------------:|
| 内存使用 | 高（一次性处理所有图片） | 低（分块处理） |
| 适用场景 | 短序列、小场景 | 长序列、大场景 |
| 输出格式 | 高斯模型（PLY/视频） | 点云、相机姿态 |
| 处理速度 | 快（单次推理） | 中等（分块+对齐） |
| 精度 | 高 | 高（带回环优化） |

## 故障排除

### 问题1: 仍然内存不足

**解决方案**:
- 进一步降低 `chunk_size`（如果已经是30，可能需要更小的值，需要修改代码）
- 降低处理分辨率（修改配置文件中的分辨率设置）
- 使用 CPU 模式（会很慢）

### 问题2: 找不到 faiss 模块

```bash
pip install faiss-cpu
# 或
pip install faiss-gpu
```

### 问题3: 找不到权重文件

```bash
bash da3_streaming/scripts/download_weights.sh
```

### 问题4: 磁盘空间不足

- 使用 `--no-save-depth-conf` 不保存深度结果
- 使用 `--delete-temp` 自动删除临时文件
- 确保有足够的磁盘空间（长序列可能需要50GB+）

## 性能参考

根据官方测试（KITTI序列，A100 GPU）：

| 方法 | 时间 (11,373帧) | FPS |
|:----:|:--------------:|:---:|
| VGGT-Long | 65min 08sec | 2.91 |
| Pi-Long | 60min 09sec | 3.15 |
| **DA3-Streaming** | **22min 17sec** | **8.51** |

## 更多信息

- DA3-Streaming README: `da3_streaming/README.md`
- 项目主 README: `README.md`
- GitHub: https://github.com/ByteDance-Seed/Depth-Anything-3

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间，长序列可能需要50GB+
2. **处理时间**: 分块处理需要更多时间，但内存使用大幅降低
3. **精度**: DA3-Streaming 通过回环检测和分块对齐保证精度
4. **输出格式**: DA3-Streaming 主要输出点云，不是直接的高斯模型
