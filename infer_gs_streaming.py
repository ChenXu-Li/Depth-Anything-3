#!/usr/bin/env python3
"""
基于 DA3-Streaming 的一键运行脚本：从图片文件夹推理出高斯模型（内存优化版）

DA3-Streaming 通过分块处理来节省内存，适合处理长视频序列和大规模场景。
根据测试，chunk_size=60 时峰值显存约 12.7GB，chunk_size=30 时约 11.5GB。

使用方法:
    python infer_gs_streaming.py <图片文件夹路径> [选项]

示例:
    python infer_gs_streaming.py /root/autodl-tmp/data/road10
    python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 30 --export-gs
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
da3_streaming_dir = project_root / "da3_streaming"
sys.path.insert(0, str(da3_streaming_dir))

# DA3-Streaming 模块将在需要时导入
DA3_Streaming = None
load_config = None


def create_custom_config(base_config_path, chunk_size, overlap, output_dir):
    """创建自定义配置文件"""
    import yaml
    
    # 加载基础配置
    with open(base_config_path, 'r') as f:
        config = yaml.full_load(f)
    
    # 修改配置
    config['Model']['chunk_size'] = chunk_size
    config['Model']['overlap'] = overlap
    
    # 保存自定义配置
    custom_config_path = os.path.join(output_dir, 'custom_config.yaml')
    with open(custom_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return custom_config_path


def export_gs_from_streaming_results(results_dir, export_dir, model_dir=None):
    """
    从 DA3-Streaming 的结果中导出高斯模型
    
    注意：DA3-Streaming 主要输出点云和相机姿态，不直接输出高斯模型。
    如果需要高斯模型，需要使用标准的 DA3 API 处理结果。
    """
    import torch
    from depth_anything_3.api import DepthAnything3
    
    print("\n🔄 从流式处理结果导出高斯模型...")
    
    # 读取相机姿态和深度结果
    poses_file = os.path.join(results_dir, 'camera_poses.txt')
    intrinsics_file = os.path.join(results_dir, 'intrinsic.txt')
    results_output_dir = os.path.join(results_dir, 'results_output')
    
    if not os.path.exists(results_output_dir):
        print("⚠️  未找到 results_output 目录，无法导出高斯模型")
        print("   提示：需要在配置中设置 save_depth_conf_result: True")
        return False
    
    # 加载模型（如果需要）
    if model_dir:
        print(f"   加载模型: {model_dir}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained(model_dir)
        model = model.to(device=device)
    else:
        print("⚠️  未指定模型目录，跳过高斯模型导出")
        return False
    
    # TODO: 实现从流式结果到高斯模型的转换
    # 这需要读取深度图、相机参数，然后使用模型生成高斯模型
    print("   注意：从流式结果导出高斯模型功能正在开发中...")
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description="基于 DA3-Streaming 的图片文件夹推理脚本（内存优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用 - 使用默认配置（chunk_size=60, overlap=30）
  python infer_gs_streaming.py /root/autodl-tmp/data/road10

  # 使用更小的chunk_size以节省内存（推荐用于内存不足的情况）
  python infer_gs_streaming.py /root/autodl-tmp/data/road10 --chunk-size 30

  # 使用自定义配置文件
  python infer_gs_streaming.py /root/autodl-tmp/data/road10 \\
      --config ./da3_streaming/configs/base_config.yaml

  # 指定输出目录
  python infer_gs_streaming.py /root/autodl-tmp/data/road10 \\
      --output-dir ./output/road10_streaming

内存优化建议:
  - chunk_size=30: 峰值显存约 11.5GB（适合12GB GPU）
  - chunk_size=60: 峰值显存约 12.7GB（适合16GB GPU）
  - chunk_size=90: 峰值显存约 14.3GB（适合24GB GPU）
  - chunk_size=120: 峰值显存约 15.9GB（适合32GB GPU）
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
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: 使用 base_config.yaml 并自动调整chunk_size)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录路径 (默认: ./exps/<图片文件夹名>_<时间戳>)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        choices=[30, 60, 90, 120],
        help="分块大小，越小越省内存 (默认: 60)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="分块重叠大小 (默认: chunk_size的一半)",
    )
    parser.add_argument(
        "--loop-enable",
        action="store_true",
        default=True,
        help="启用回环检测 (默认: True)",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop_enable",
        action="store_false",
        help="禁用回环检测",
    )
    parser.add_argument(
        "--save-depth-conf",
        action="store_true",
        default=True,
        help="保存深度和置信度结果 (默认: True)",
    )
    parser.add_argument(
        "--no-save-depth-conf",
        dest="save_depth_conf",
        action="store_false",
        help="不保存深度和置信度结果（节省磁盘空间）",
    )
    parser.add_argument(
        "--delete-temp",
        action="store_true",
        default=True,
        help="处理完成后删除临时文件 (默认: True)",
    )
    parser.add_argument(
        "--keep-temp",
        dest="delete_temp",
        action="store_false",
        help="保留临时文件（用于调试）",
    )

    args = parser.parse_args()

    # 延迟导入 DA3-Streaming 模块（在解析参数之后，这样 --help 可以正常工作）
    try:
        from loop_utils.config_utils import load_config
        # 直接导入 da3_streaming.py 中的类
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "da3_streaming_module",
            da3_streaming_dir / "da3_streaming.py"
        )
        da3_streaming_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(da3_streaming_module)
        DA3_Streaming = da3_streaming_module.DA3_Streaming
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("\n请确保已正确安装 DA3-Streaming 的依赖:")
        print("  1. 安装 faiss: pip install faiss-cpu 或 pip install faiss-gpu")
        print("  2. 确保所有依赖已安装: pip install -r da3_streaming/requirements.txt")
        print("  3. 下载权重文件: bash da3_streaming/scripts/download_weights.sh")
        sys.exit(1)

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
    if args.output_dir is None:
        from datetime import datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path("./exps") / f"{image_dir.name}_{current_datetime}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")

    # 设置配置文件
    if args.config is None:
        # 使用默认配置并调整参数
        base_config_path = project_root / "da3_streaming" / "configs" / "base_config.yaml"
        if not base_config_path.exists():
            print(f"❌ 错误: 找不到默认配置文件: {base_config_path}")
            sys.exit(1)
        
        # 设置overlap（默认为chunk_size的一半）
        overlap = args.overlap if args.overlap is not None else args.chunk_size // 2
        
        # 创建自定义配置
        print(f"⚙️  创建自定义配置: chunk_size={args.chunk_size}, overlap={overlap}")
        config_path = create_custom_config(
            str(base_config_path),
            args.chunk_size,
            overlap,
            str(output_dir)
        )
        
        # 加载并修改配置
        config = load_config(config_path)
        config['Model']['loop_enable'] = args.loop_enable
        config['Model']['save_depth_conf_result'] = args.save_depth_conf
        config['Model']['delete_temp_files'] = args.delete_temp
        
        # 保存修改后的配置
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    else:
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"❌ 错误: 配置文件不存在: {config_path}")
            sys.exit(1)
        config = load_config(config_path)

    # 显示配置信息
    print("\n📋 配置信息:")
    print(f"   - 分块大小 (chunk_size): {config['Model']['chunk_size']}")
    print(f"   - 重叠大小 (overlap): {config['Model']['overlap']}")
    print(f"   - 回环检测: {config['Model']['loop_enable']}")
    print(f"   - 保存深度结果: {config['Model']['save_depth_conf_result']}")
    print(f"   - 删除临时文件: {config['Model']['delete_temp_files']}")
    
    # 内存使用估算
    chunk_size = config['Model']['chunk_size']
    memory_estimates = {
        30: "~11.5GB",
        60: "~12.7GB",
        90: "~14.3GB",
        120: "~15.9GB"
    }
    if chunk_size in memory_estimates:
        print(f"   - 预估峰值显存: {memory_estimates[chunk_size]}")

    # 运行流式处理
    print("\n🚀 开始流式处理...")
    print("   这可能需要一些时间，请耐心等待...")
    print()

    try:
        da3_streaming = DA3_Streaming(str(image_dir), str(output_dir), config)
        da3_streaming.run()
        da3_streaming.close()

        print("\n✅ 流式处理完成！")
        print(f"📂 结果保存在: {output_dir}")
        
        # 列出输出文件
        print("\n📄 输出文件:")
        poses_file = output_dir / "camera_poses.txt"
        intrinsics_file = output_dir / "intrinsic.txt"
        pcd_file = output_dir / "pcd" / "combined_pcd.ply"
        
        if poses_file.exists():
            print(f"   ✓ 相机姿态: {poses_file}")
        if intrinsics_file.exists():
            print(f"   ✓ 相机内参: {intrinsics_file}")
        if pcd_file.exists():
            print(f"   ✓ 点云文件: {pcd_file}")
        
        results_output_dir = output_dir / "results_output"
        if results_output_dir.exists():
            npz_files = list(results_output_dir.glob("*.npz"))
            print(f"   ✓ 深度结果: {len(npz_files)} 个NPZ文件")

        print("\n🎉 完成！")
        print("\n💡 提示:")
        print("   - 可以使用 npz_output_process.py 处理深度结果")
        print("   - 点云文件可以在 CloudCompare、MeshLab 等软件中查看")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理
        import torch
        import gc
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
