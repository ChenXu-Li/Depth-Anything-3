# 一键推理高斯模型脚本使用说明

## 脚本功能

`infer_gs_from_images.py` 是一个一键运行脚本，可以从图片文件夹中自动推理出3D高斯模型（3D Gaussian Splatting）。

## 快速开始

### 基本使用

```bash
# 从图片文件夹推理并导出PLY格式的高斯模型
python infer_gs_from_images.py /root/autodl-tmp/data/road10
```

### 导出视频格式

```bash
# 导出渲染视频（需要安装gsplat依赖）
python infer_gs_from_images.py /root/autodl-tmp/data/road10 --export-format gs_video
```

## 完整参数说明

### 必需参数

- `image_dir`: 包含图片的文件夹路径

### 可选参数

#### 模型相关
- `--model-dir`: 模型目录路径或Hugging Face仓库名
  - 默认: `depth-anything/DA3NESTED-GIANT-LARGE-1.1`
  - 示例: `--model-dir depth-anything/DA3-GIANT-1.1`

#### 输出相关
- `--export-dir`: 输出目录路径
  - 默认: `<图片文件夹>_gs_output`
  - 示例: `--export-dir ./output/road10`

- `--export-format`: 导出格式
  - 选项: `gs_ply` (PLY点云文件) 或 `gs_video` (渲染视频)
  - 默认: `gs_ply`

#### 处理参数
- `--device`: 使用的设备
  - 选项: `cuda` 或 `cpu`
  - 默认: `cuda`

- `--process-res`: 处理分辨率
  - 默认: `504`

- `--process-res-method`: 处理分辨率方法
  - 选项: `upper_bound_resize`, `lower_bound_resize`, `exact`
  - 默认: `upper_bound_resize`

#### 姿态估计参数
- `--use-ray-pose`: 使用基于射线的姿态估计（更准确但更慢）
- `--ref-view-strategy`: 参考视图选择策略
  - 选项: `first`, `middle`, `saddle_balanced`, `saddle_sim_range`
  - 默认: `saddle_balanced`

#### 视频导出参数（仅当 `--export-format gs_video` 时）
- `--gs-trj-mode`: 渲染轨迹模式
  - 选项: `smooth`, `linear`
  - 默认: `smooth`

- `--gs-video-quality`: 视频质量
  - 选项: `low`, `medium`, `high`
  - 默认: `medium`

## 使用示例

### 示例1: 基本推理（PLY格式）

```bash
python infer_gs_from_images.py /root/autodl-tmp/data/road10
```

输出：
- 在 `/root/autodl-tmp/data/road10_gs_output/gs_ply/` 目录下生成PLY文件

### 示例2: 导出视频格式

```bash
python infer_gs_from_images.py /root/autodl-tmp/data/road10 \
    --export-format gs_video \
    --gs-video-quality high
```

输出：
- 在 `/root/autodl-tmp/data/road10_gs_output/gs_video/` 目录下生成MP4视频文件

### 示例3: 使用自定义模型和输出目录

```bash
python infer_gs_from_images.py /root/autodl-tmp/data/road10 \
    --model-dir depth-anything/DA3-GIANT-1.1 \
    --export-dir ./my_output/road10 \
    --process-res 1024
```

### 示例4: 使用更准确的姿态估计

```bash
python infer_gs_from_images.py /root/autodl-tmp/data/road10 \
    --use-ray-pose \
    --ref-view-strategy saddle_balanced
```

## 输出文件说明

### PLY格式 (`gs_ply`)
- 位置: `<export_dir>/gs_ply/`
- 文件: `0000.ply`, `0001.ply`, ... (每个视图一个PLY文件)
- 用途: 可以在3D软件中查看和编辑点云

### 视频格式 (`gs_video`)
- 位置: `<export_dir>/gs_video/`
- 文件: `*.mp4` (渲染的新视角视频)
- 用途: 查看新视角渲染效果

## 注意事项

1. **GPU要求**: 推荐使用CUDA设备，CPU推理会很慢
2. **内存要求**: 大分辨率图片需要更多GPU内存
3. **gsplat依赖**: 导出视频格式需要安装gsplat:
   ```bash
   pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
   ```
4. **模型下载**: 首次使用会自动从Hugging Face下载模型，需要网络连接

## 故障排除

### 问题1: CUDA内存不足
- 降低 `--process-res` 参数（如改为256或384）
- 使用 `--device cpu`（会很慢）

### 问题2: 找不到图片文件
- 确保文件夹中包含 `.jpg`, `.jpeg`, `.png` 等格式的图片
- 检查文件夹路径是否正确

### 问题3: 模型下载失败
- 检查网络连接
- 可以设置镜像: `export HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载模型到本地，然后使用 `--model-dir` 指定本地路径

## 技术支持

如有问题，请参考：
- 项目README: `README.md`
- CLI文档: `docs/CLI.md`
- API文档: `docs/API.md`
