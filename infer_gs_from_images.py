#!/usr/bin/env python3
"""
一键运行脚本：从图片文件夹推理出高斯模型（3D Gaussian Splatting）

使用方法:
    python infer_gs_from_images.py <图片文件夹路径> [选项]

示例:
    python infer_gs_from_images.py /root/autodl-tmp/data/road10
    python infer_gs_from_images.py /root/autodl-tmp/data/road10 --export-format gs_video
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser(
        description="从图片文件夹推理出高斯模型（3D Gaussian Splatting）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用 - 导出PLY格式
  python infer_gs_from_images.py /root/autodl-tmp/ExtractColmapImages/extracted_images

  # 导出视频格式
  python infer_gs_from_images.py /root/autodl-tmp/data/road10 --export-format gs_video

  # 指定输出目录和模型
  python infer_gs_from_images.py /root/autodl-tmp/data/road10 \\
      --export-dir ./output/road10 \\
      --model-dir depth-anything/DA3NESTED-GIANT-LARGE-1.1
        """,
    )

    # 必需参数
    parser.add_argument(
        "image_dir",
        type=str,
        help="包含图片的文件夹路径",
    )

    # 可选参数
    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="模型目录路径或Hugging Face仓库名 (默认: depth-anything/DA3NESTED-GIANT-LARGE-1.1)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="输出目录路径 (默认: <图片文件夹>_gs_output)",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="gs_ply",
        choices=["gs_ply", "gs_video"],
        help="导出格式: gs_ply (PLY点云文件) 或 gs_video (渲染视频) (默认: gs_ply)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="使用的设备 (默认: cuda)",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="处理分辨率 (默认: 504)",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize", "exact"],
        help="处理分辨率方法 (默认: upper_bound_resize)",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="使用基于射线的姿态估计（更准确但更慢）",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="参考视图选择策略 (默认: saddle_balanced)",
    )

    # GS视频导出参数
    parser.add_argument(
        "--gs-trj-mode",
        type=str,
        default="smooth",
        choices=["smooth", "linear"],
        help="[gs_video] 渲染轨迹模式 (默认: smooth)",
    )
    parser.add_argument(
        "--gs-video-quality",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="[gs_video] 视频质量 (默认: medium)",
    )

    args = parser.parse_args()

    # 检查输入目录
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"❌ 错误: 图片文件夹不存在: {image_dir}")
        sys.exit(1)

    if not image_dir.is_dir():
        print(f"❌ 错误: 路径不是文件夹: {image_dir}")
        sys.exit(1)

    # 查找图片文件
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(image_dir / ext)))
        image_files.extend(glob.glob(str(image_dir / ext.upper())))

    if not image_files:
        print(f"❌ 错误: 在 {image_dir} 中未找到图片文件")
        print(f"   支持的格式: {', '.join(image_extensions)}")
        sys.exit(1)

    # 排序图片文件
    image_files = sorted(image_files)
    print(f"✅ 找到 {len(image_files)} 张图片")

    # 设置输出目录
    if args.export_dir is None:
        export_dir = image_dir.parent / f"{image_dir.name}_gs_output"
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
    print("\n🚀 开始推理...")
    print(f"   - 图片数量: {len(image_files)}")
    print(f"   - 处理分辨率: {args.process_res}")
    print(f"   - 导出格式: {args.export_format}")
    print(f"   - 启用3D高斯模型: True")
    print()

    try:
        prediction = model.inference(
            image=image_files,
            export_dir=str(export_dir),
            export_format=args.export_format,
            infer_gs=True,  # 启用高斯模型
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            use_ray_pose=args.use_ray_pose,
            ref_view_strategy=args.ref_view_strategy,
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
