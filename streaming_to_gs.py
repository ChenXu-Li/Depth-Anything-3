#!/usr/bin/env python3
"""
从 DA3-Streaming 结果生成高斯模型脚本

该脚本读取 DA3-Streaming 的输出结果（相机姿态、深度等），
然后使用标准 DA3 API 重新推理并生成高斯模型。

使用方法:
    python streaming_to_gs.py <streaming_output_dir> [选项]

示例:
    python streaming_to_gs.py exps/road10_2026-02-28-23-32-27
    python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 --export-format gs_video
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from depth_anything_3.api import DepthAnything3


def load_camera_poses(poses_file):
    """加载相机姿态文件"""
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    return np.array(poses)


def load_intrinsics(intrinsics_file):
    """加载相机内参文件"""
    intrinsics = []
    with open(intrinsics_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 4:
                fx, fy, cx, cy = values
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                intrinsics.append(K)
    return np.array(intrinsics)


def load_images_from_npz(npz_dir):
    """从NPZ文件加载图片"""
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    images = []
    for npz_file in npz_files:
        data = np.load(npz_file)
        if 'image' in data:
            images.append(data['image'])
    return images, npz_files


def find_original_images(streaming_output_dir):
    """查找原始图片路径（从results_output中的NPZ文件名推断）"""
    npz_dir = os.path.join(streaming_output_dir, "results_output")
    if not os.path.exists(npz_dir):
        return None
    
    # 尝试从NPZ文件名推断原始图片路径
    # 通常NPZ文件名格式为 frame_<index>.npz
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        return None
    
    # 读取第一个NPZ文件，查看是否有原始路径信息
    # 如果没有，需要用户提供原始图片目录
    return None


def main():
    parser = argparse.ArgumentParser(
        description="从 DA3-Streaming 结果生成高斯模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用 - 需要提供原始图片目录
  python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 \\
      --image-dir /root/autodl-tmp/data/road10

  # 导出视频格式
  python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 \\
      --image-dir /root/autodl-tmp/data/road10 \\
      --export-format gs_video

  # 使用已有的相机姿态和内参（从NPZ文件读取）
  python streaming_to_gs.py exps/road10_2026-02-28-23-32-27 \\
      --image-dir /root/autodl-tmp/data/road10 \\
      --use-streaming-poses
        """,
    )

    # 必需参数
    parser.add_argument(
        "streaming_output_dir",
        type=str,
        help="DA3-Streaming 的输出目录路径",
    )

    # 可选参数
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="原始图片文件夹路径（如果使用--use-streaming-poses，可以从NPZ文件读取）",
    )
    parser.add_argument(
        "--use-streaming-poses",
        action="store_true",
        help="使用流式处理得到的相机姿态和内参（从NPZ文件读取）",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="模型目录路径或Hugging Face仓库名",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="输出目录路径 (默认: <streaming_output_dir>/gs_output)",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="gs_ply",
        choices=["gs_ply", "gs_video"],
        help="导出格式: gs_ply (PLY点云文件) 或 gs_video (渲染视频)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用的设备",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="处理分辨率",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="使用基于射线的姿态估计（更准确但更慢）",
    )

    # GS视频导出参数
    parser.add_argument(
        "--gs-trj-mode",
        type=str,
        default="smooth",
        choices=["smooth", "linear", "extend"],
        help="[gs_video] 渲染轨迹模式",
    )
    parser.add_argument(
        "--gs-video-quality",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="[gs_video] 视频质量",
    )

    args = parser.parse_args()

    # 检查输入目录
    streaming_output_dir = Path(args.streaming_output_dir)
    if not streaming_output_dir.exists():
        print(f"❌ 错误: 流式输出目录不存在: {streaming_output_dir}")
        sys.exit(1)

    # 检查必要文件
    poses_file = streaming_output_dir / "camera_poses.txt"
    intrinsics_file = streaming_output_dir / "intrinsic.txt"
    npz_dir = streaming_output_dir / "results_output"

    if not poses_file.exists():
        print(f"❌ 错误: 找不到相机姿态文件: {poses_file}")
        sys.exit(1)

    if not intrinsics_file.exists():
        print(f"❌ 错误: 找不到相机内参文件: {intrinsics_file}")
        sys.exit(1)

    # 自动检测：如果NPZ目录存在且有文件，优先使用NPZ；否则需要提供image_dir
    use_npz = args.use_streaming_poses
    if not use_npz and args.image_dir is None:
        # 自动检测NPZ文件
        if npz_dir.exists():
            npz_files = glob.glob(str(npz_dir / "*.npz"))
            if npz_files:
                print("ℹ️  检测到NPZ文件，自动使用流式处理结果")
                use_npz = True
    
    # 确定图片路径
    if use_npz:
        # 从NPZ文件读取图片
        if not npz_dir.exists():
            print(f"❌ 错误: 找不到NPZ结果目录: {npz_dir}")
            print("   提示: 需要在流式处理时启用 save_depth_conf_result")
            sys.exit(1)
        
        images, npz_files = load_images_from_npz(str(npz_dir))
        if not images:
            print(f"❌ 错误: 在 {npz_dir} 中未找到NPZ文件")
            sys.exit(1)
        
        print(f"✅ 从NPZ文件加载了 {len(images)} 张图片")
        
        # 加载相机姿态和内参
        poses = load_camera_poses(str(poses_file))
        intrinsics = load_intrinsics(str(intrinsics_file))
        
        if len(poses) != len(images) or len(intrinsics) != len(images):
            print(f"⚠️  警告: 图片数量({len(images)})与姿态数量({len(poses)})或内参数量({len(intrinsics)})不匹配")
            min_len = min(len(images), len(poses), len(intrinsics))
            images = images[:min_len]
            poses = poses[:min_len]
            intrinsics = intrinsics[:min_len]
            print(f"   使用前 {min_len} 个数据")
        
        # 转换姿态格式：从C2W转换为W2C（OpenCV格式）
        # API期望4x4矩阵，所以保留完整的4x4矩阵
        w2c_poses = []
        for c2w in poses:
            w2c = np.linalg.inv(c2w)
            w2c_poses.append(w2c)  # 保留完整的4x4矩阵
        extrinsics = np.array(w2c_poses)
        
        image_paths = None  # 使用numpy数组而不是文件路径
        
    else:
        # 使用原始图片文件
        if args.image_dir is None:
            print("❌ 错误: 需要提供 --image-dir 或确保流式处理时保存了NPZ文件")
            print("   提示:")
            print("   1. 使用 --image-dir 指定原始图片目录")
            print("   2. 或在流式处理时使用 --save-depth-conf 保存NPZ文件")
            sys.exit(1)
        
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"❌ 错误: 图片文件夹不存在: {image_dir}")
            sys.exit(1)
        
        # 查找图片文件
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(image_dir / ext)))
            image_files.extend(glob.glob(str(image_dir / ext.upper())))
        
        if not image_files:
            print(f"❌ 错误: 在 {image_dir} 中未找到图片文件")
            sys.exit(1)
        
        image_files = sorted(image_files)
        print(f"✅ 找到 {len(image_files)} 张图片")
        
        # 加载相机姿态和内参
        poses = load_camera_poses(str(poses_file))
        intrinsics = load_intrinsics(str(intrinsics_file))
        
        if len(poses) != len(image_files) or len(intrinsics) != len(image_files):
            print(f"⚠️  警告: 图片数量({len(image_files)})与姿态数量({len(poses)})或内参数量({len(intrinsics)})不匹配")
            min_len = min(len(image_files), len(poses), len(intrinsics))
            image_files = image_files[:min_len]
            poses = poses[:min_len]
            intrinsics = intrinsics[:min_len]
            print(f"   使用前 {min_len} 个数据")
        
        # 转换姿态格式：从C2W转换为W2C（OpenCV格式）
        # API期望4x4矩阵，所以保留完整的4x4矩阵
        w2c_poses = []
        for c2w in poses:
            w2c = np.linalg.inv(c2w)
            w2c_poses.append(w2c)  # 保留完整的4x4矩阵
        extrinsics = np.array(w2c_poses)
        
        image_paths = image_files

    # 设置输出目录
    if args.export_dir is None:
        export_dir = streaming_output_dir / "gs_output"
    else:
        export_dir = Path(args.export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {export_dir}")

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = "cpu"

    # 加载模型
    print(f"🔄 加载模型: {args.model_dir}")
    print("   这可能需要一些时间，请耐心等待...")
    try:
        device = torch.device(args.device)
        model = DepthAnything3.from_pretrained(args.model_dir)
        model = model.to(device=device)
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # 准备导出参数
    export_kwargs = {}
    if args.export_format == "gs_video":
        export_kwargs = {
            "gs_video": {
                "trj_mode": args.gs_trj_mode,
                "video_quality": args.gs_video_quality,
            }
        }

    # 运行推理
    print("\n🚀 开始推理并生成高斯模型...")
    print(f"   - 图片数量: {len(images) if image_paths is None else len(image_paths)}")
    print(f"   - 处理分辨率: {args.process_res}")
    print(f"   - 导出格式: {args.export_format}")
    print(f"   - 启用3D高斯模型: True")
    print()

    try:
        # 准备输入
        if image_paths is None:
            # 使用numpy数组
            input_images = images
        else:
            # 使用文件路径
            input_images = image_paths

        # 运行推理（使用已有的相机姿态和内参）
        prediction = model.inference(
            image=input_images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_dir=str(export_dir),
            export_format=args.export_format,
            infer_gs=True,  # 启用高斯模型
            process_res=args.process_res,
            use_ray_pose=args.use_ray_pose,
            align_to_input_ext_scale=False,  # 使用提供的姿态，不重新对齐
            export_kwargs=export_kwargs,
        )

        print("\n✅ 推理完成！")
        print(f"📂 结果保存在: {export_dir}")

        # 列出生成的文件
        if args.export_format == "gs_ply":
            gs_ply_dir = export_dir / "gs_ply"
            if gs_ply_dir.exists():
                ply_files = list(gs_ply_dir.glob("*.ply"))
                print(f"   - PLY文件数量: {len(ply_files)}")
                if ply_files:
                    print(f"   - 示例文件: {ply_files[0].name}")
        elif args.export_format == "gs_video":
            gs_video_dir = export_dir / "gs_video"
            if gs_video_dir.exists():
                video_files = list(gs_video_dir.glob("*.mp4"))
                print(f"   - 视频文件数量: {len(video_files)}")
                if video_files:
                    print(f"   - 示例文件: {video_files[0].name}")

        print("\n🎉 完成！")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
