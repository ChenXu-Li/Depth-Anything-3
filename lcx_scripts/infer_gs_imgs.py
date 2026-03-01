#!/usr/bin/env python3
"""
一键运行脚本：从图片文件夹推理出高斯模型（3D Gaussian Splatting）

改进版特性：
- 自动半精度 (CUDA)
- 自动 CUDA fallback
- 更稳健的图片扫描
- 保存运行配置
- 更好的异常处理
python infer_gs_imgs.py /root/autodl-tmp/ExtractColmapImages/extracted_images
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from depth_anything_3.api import DepthAnything3


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(image_dir: Path):
    return sorted(
        [str(p) for p in image_dir.rglob("*")
         if p.suffix.lower() in SUPPORTED_EXTS]
    )


def main():
    parser = argparse.ArgumentParser(
        description="从图片文件夹推理出高斯模型（3D Gaussian Splatting）"
    )

    parser.add_argument("image_dir", type=str, help="图片文件夹路径")

    parser.add_argument(
        "--model-dir",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="模型路径或 HuggingFace 仓库名",
    )

    parser.add_argument("--export-dir", type=str, default=None)

    parser.add_argument(
        "--export-format",
        type=str,
        default="gs_ply",
        choices=["gs_ply", "gs_video"],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize", "exact"],
    )

    parser.add_argument("--use-ray-pose", action="store_true")

    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
    )

    parser.add_argument("--gs-trj-mode", type=str, default="smooth",
                        choices=["smooth", "linear"])
    parser.add_argument("--gs-video-quality", type=str, default="medium",
                        choices=["low", "medium", "high"])

    args = parser.parse_args()

    image_dir = Path(args.image_dir)

    if not image_dir.exists() or not image_dir.is_dir():
        print(f"❌ 图片目录无效: {image_dir}")
        sys.exit(1)

    image_files = find_images(image_dir)

    if not image_files:
        print(f"❌ 未找到图片文件")
        sys.exit(1)

    print(f"✅ 找到 {len(image_files)} 张图片")

    # 输出目录
    if args.export_dir is None:
        export_dir = image_dir.parent / f"{image_dir.name}_gs_output"
    else:
        export_dir = Path(args.export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {export_dir}")

    # 保存运行配置
    with open(export_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 设备检查
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，自动切换 CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    # 自动混合精度
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"🔄 加载模型: {args.model_dir}")
    print(f"   设备: {device} | dtype: {dtype}")

    try:
        model = DepthAnything3.from_pretrained(
            args.model_dir,
            torch_dtype=dtype
        ).to(device)

        model.eval()

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # 视频导出参数
    export_kwargs = {}
    if args.export_format == "gs_video":
        export_kwargs = {
            "gs_video": {
                "trj_mode": args.gs_trj_mode,
                "video_quality": args.gs_video_quality,
            }
        }

    print("\n🚀 开始推理")
    print(f"   分辨率: {args.process_res}")
    print(f"   导出格式: {args.export_format}")
    print(f"   use_ray_pose: {args.use_ray_pose}")
    print()

    try:
        with torch.inference_mode():
            prediction = model.inference(
                image=image_files,
                export_dir=str(export_dir),
                export_format=args.export_format,
                infer_gs=True,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                use_ray_pose=args.use_ray_pose,
                ref_view_strategy=args.ref_view_strategy,
                export_kwargs=export_kwargs,
            )

        print("\n✅ 推理完成")
        print(f"📂 结果保存至: {export_dir}")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n❌ CUDA 显存不足")
            print("👉 建议降低 --process-res 或使用 --device cpu")
        else:
            print(f"\n❌ 运行错误: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        sys.exit(1)

    print("\n🎉 完成！")


if __name__ == "__main__":
    main()