import os
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
import glob
from PIL import Image
import pycolmap
import re
from collections import defaultdict


def load_rig_colmap_data_with_skip(
    colmap_dir: str, 
    sparse_subdir: str = "0",
    skip_step: int = 1,  # 跳过步长：1=使用所有图片，2=每隔1张取1张，依此类推
    image_indices: list[int] = None,  # 图片索引列表（排序后的第几张，从0开始）
    camera_indices: list[int] = None  # 相机索引列表（pano_camera编号）
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    从Rig COLMAP目录加载图片路径和相机参数，并按步长跳过图片
    支持Rig格式的COLMAP数据（包含rigs.bin和frames.bin）
    支持根据硬编码的图片索引和相机索引列表进行视角选择
    
    Args:
        colmap_dir: COLMAP根目录（包含images/和sparse/子目录）
        sparse_subdir: COLMAP稀疏重建子目录（如"0"对应sparse/0/）
        skip_step: 图片采样步长，1=全部使用，n=每n张取1张
        image_indices: 图片索引列表（排序后的第几张，从0开始），None表示使用所有图片
        camera_indices: 相机索引列表（pano_camera编号），None表示使用所有相机
    
    Returns:
        image_paths: 采样后的图片路径列表
        extrinsics: 对应采样图片的外参矩阵 (N, 4, 4) - world to camera transformation
        intrinsics: 对应采样图片的内参矩阵 (N, 3, 3)
    """
    colmap_dir = Path(colmap_dir)
    images_dir = colmap_dir / "images"
    sparse_dir = colmap_dir / "sparse" / sparse_subdir
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not sparse_dir.exists():
        raise ValueError(f"Sparse reconstruction directory not found: {sparse_dir}")
    
    # 检查是否为rig格式（存在rigs.bin和frames.bin）
    has_rigs = (sparse_dir / "rigs.bin").exists()
    has_frames = (sparse_dir / "frames.bin").exists()
    is_rig_format = has_rigs and has_frames
    
    if is_rig_format:
        print(f"检测到Rig COLMAP格式（包含rigs.bin和frames.bin）")
    else:
        print(f"警告：未检测到rig格式文件，将按标准COLMAP格式处理")
    
    # 加载COLMAP重建
    try:
        recon = pycolmap.Reconstruction(str(sparse_dir))
    except Exception as e:
        raise RuntimeError(f"Failed to load COLMAP reconstruction from {sparse_dir}: {e}")
    
    # 收集图像数据（先按相机分组，以便确定图片索引）
    # 使用字典按相机分组：{camera_idx: [(image_id, image, image_name, ...), ...]}
    images_by_camera = defaultdict(list)
    
    # 按图像ID排序以确保顺序一致
    sorted_images = sorted(recon.images.items(), key=lambda x: x[0])
    
    # 解析相机编号的正则表达式
    camera_pattern = re.compile(r'pano_camera(\d+)')
    
    # 第一遍：按相机分组收集所有图片
    for image_id, image in sorted_images:
        image_name = image.name
        # 从路径中提取相机编号
        camera_idx = None
        if '/' in image_name:
            # 路径格式：pano_camera0/xxx.jpg
            match = camera_pattern.search(image_name)
            if match:
                camera_idx = int(match.group(1))
        else:
            # 如果没有子目录，尝试从images目录结构推断
            # 这种情况下可能需要直接检查文件系统
            pass
        
        # 如果无法从路径解析，尝试从文件系统推断
        if camera_idx is None:
            # 尝试直接检查文件路径
            potential_path = images_dir / image_name
            if potential_path.exists():
                # 检查父目录是否是pano_camera格式
                parent_dir = potential_path.parent.name
                match = camera_pattern.search(parent_dir)
                if match:
                    camera_idx = int(match.group(1))
        
        # 如果仍然无法确定，跳过或使用默认值
        if camera_idx is None:
            print(f"Warning: Cannot determine camera index for image {image_name}, skipping...")
            continue
        
        images_by_camera[camera_idx].append((image_id, image, image_name))
    
    # 第二遍：对每个相机的图片进行排序，确定图片索引，并根据硬编码列表过滤
    image_paths = []
    extrinsics_list = []
    intrinsics_list = []
    
    # 确定要使用的相机和图片索引
    if camera_indices is None:
        # 使用所有相机
        selected_cameras = sorted(images_by_camera.keys())
    else:
        # 使用指定的相机
        selected_cameras = sorted(set(camera_indices))
        print(f"使用指定的相机索引: {selected_cameras}")
    
    if image_indices is None:
        # 使用所有图片索引（将在后续通过skip_step采样）
        selected_image_indices = None
    else:
        # 使用指定的图片索引
        selected_image_indices = set(image_indices)
        print(f"使用指定的图片索引: {sorted(selected_image_indices)}")
    
    # 遍历选定的相机
    for camera_idx in selected_cameras:
        if camera_idx not in images_by_camera:
            print(f"Warning: Camera {camera_idx} not found in reconstruction, skipping...")
            continue
        
        # 获取该相机的所有图片并排序（按文件名）
        camera_images = images_by_camera[camera_idx]
        # 按图片名称排序，确保索引一致性
        camera_images_sorted = sorted(camera_images, key=lambda x: x[2])  # 按image_name排序
        
        # 遍历该相机的所有图片
        for img_idx_in_camera, (image_id, image, image_name) in enumerate(camera_images_sorted):
            # 如果指定了图片索引列表，检查当前图片索引是否在列表中
            if selected_image_indices is not None and img_idx_in_camera not in selected_image_indices:
                continue
            
            # 处理rig格式中的嵌套路径（如 images/pano_camera0/xxx.jpg）
            if '/' in image_name:
                # 如果路径中包含子目录，直接使用
                image_path = images_dir / image_name
            else:
                # 否则尝试从相机目录构建路径
                image_path = images_dir / f"pano_camera{camera_idx}" / image_name
            
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}, skipping...")
                continue
            
            image_paths.append(str(image_path.absolute()))
            
            # 获取相机
            camera = recon.cameras[image.camera_id]
            
            # 提取外参（world to camera transformation）
            # pycolmap的cam_from_world()方法应该已经能够处理rig格式
            # 它会自动从frame中获取rig_from_world，然后结合cam_from_rig计算cam_from_world
            try:
                # pycolmap的image.cam_from_world()方法应该已经处理了rig格式
                # 它会自动从frame获取rig_from_world，然后结合rig中的cam_from_rig计算
                cam_from_world = image.cam_from_world()
            except Exception as e:
                # 如果直接调用失败，输出错误信息
                print(f"Warning: Failed to get cam_from_world for image {image_id} ({image_name}): {e}")
                print(f"  This might be due to rig format handling. Trying alternative method...")
                
                # 尝试备用方法：如果image有frame_id，尝试从frame获取
                if is_rig_format and hasattr(image, 'frame_id') and image.frame_id is not None:
                    try:
                        # 尝试通过frame获取
                        if hasattr(recon, 'frame'):
                            frame = recon.frame(image.frame_id)
                            rig_from_world = frame.rig_from_world
                            # 对于rig格式，如果无法获取cam_from_rig，假设是trivial rig
                            # 即 cam_from_rig = identity，所以 cam_from_world = rig_from_world
                            cam_from_world = rig_from_world
                            print(f"  Using rig_from_world as fallback for image {image_id}")
                        else:
                            raise RuntimeError(f"Cannot access frames from reconstruction")
                    except Exception as inner_e:
                        raise RuntimeError(f"Cannot get cam_from_world for image {image_id}: {e}, fallback also failed: {inner_e}")
                else:
                    raise RuntimeError(f"Cannot get cam_from_world for image {image_id}: {e}")
            
            R = cam_from_world.rotation.matrix()  # (3, 3) rotation matrix
            t = cam_from_world.translation  # (3,) translation vector
            
            # 构建4x4外参矩阵（world to camera）
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t
            extrinsics_list.append(extrinsic)
            
            # 提取内参
            intrinsic = np.eye(3, dtype=np.float32)
            
            if camera.model == pycolmap.CameraModelId.PINHOLE:
                # PINHOLE: [fx, fy, cx, cy]
                if len(camera.params) >= 4:
                    fx, fy, cx, cy = camera.params[:4]
                    intrinsic[0, 0] = fx
                    intrinsic[1, 1] = fy
                    intrinsic[0, 2] = cx
                    intrinsic[1, 2] = cy
                elif len(camera.params) >= 3:
                    # SIMPLE_PINHOLE: [f, cx, cy]
                    f, cx, cy = camera.params[:3]
                    intrinsic[0, 0] = f
                    intrinsic[1, 1] = f
                    intrinsic[0, 2] = cx
                    intrinsic[1, 2] = cy
            elif camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                # SIMPLE_PINHOLE: [f, cx, cy]
                if len(camera.params) >= 3:
                    f, cx, cy = camera.params[:3]
                    intrinsic[0, 0] = f
                    intrinsic[1, 1] = f
                    intrinsic[0, 2] = cx
                    intrinsic[1, 2] = cy
            else:
                # 对于其他相机模型，尝试提取焦距和主点
                try:
                    if len(camera.params) >= 1:
                        f = camera.params[0]
                        intrinsic[0, 0] = f
                        intrinsic[1, 1] = f
                    if len(camera.params) >= 3:
                        cx = camera.params[-2] if len(camera.params) >= 4 else camera.params[1]
                        cy = camera.params[-1] if len(camera.params) >= 4 else camera.params[2]
                        intrinsic[0, 2] = cx
                        intrinsic[1, 2] = cy
                    else:
                        # 使用图像中心作为主点
                        intrinsic[0, 2] = camera.width / 2.0
                        intrinsic[1, 2] = camera.height / 2.0
                except Exception as e:
                    print(f"Warning: Could not extract intrinsics for camera {camera.id}, using defaults: {e}")
                    # 回退：使用图像尺寸
                    intrinsic[0, 0] = camera.width
                    intrinsic[1, 1] = camera.width
                    intrinsic[0, 2] = camera.width / 2.0
                    intrinsic[1, 2] = camera.height / 2.0
            
            intrinsics_list.append(intrinsic)
    
    if len(image_paths) == 0:
        raise RuntimeError("No valid images found in COLMAP reconstruction")
    
    # 转换为numpy数组
    full_extrinsics = np.array(extrinsics_list, dtype=np.float32)  # (N, 4, 4)
    full_intrinsics = np.array(intrinsics_list, dtype=np.float32)  # (N, 3, 3)
    
    # 按步长采样（确保至少保留1张图片）
    if skip_step < 1:
        raise ValueError("skip_step必须≥1（1表示使用所有图片）")
    
    # 采样索引：0, skip_step, 2*skip_step...
    sample_indices = list(range(0, len(image_paths), skip_step))
    if not sample_indices:  # 极端情况保护
        sample_indices = [0]
    
    # 提取采样后的数据集
    sampled_image_paths = [image_paths[i] for i in sample_indices]
    sampled_extrinsics = full_extrinsics[sample_indices]
    sampled_intrinsics = full_intrinsics[sample_indices]
    
    print(f"Rig COLMAP数据加载完成：")
    print(f"  过滤后图片数量: {len(image_paths)}")
    if image_indices is not None:
        print(f"  使用图片索引: {sorted(image_indices)}")
    if camera_indices is not None:
        print(f"  使用相机索引: {sorted(camera_indices)}")
    print(f"  采样后图片数量: {len(sampled_image_paths)} (步长={skip_step})")
    
    return sampled_image_paths, sampled_extrinsics, sampled_intrinsics

def run_rig_colmap_inference_with_skip(
    colmap_dir: str,
    skip_step: int = 1,
    model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    export_dir: str = "./rig_colmap_output",
    export_format: str = "mini_npz-glb",
    device: str = "cuda",
    image_indices: list[int] = None,  # 图片索引列表（排序后的第几张，从0开始）
    camera_indices: list[int] = None,  # 相机索引列表（pano_camera编号）
    pandepth_indices: list[int] = None,  # 用于 gs_pandepth 导出的视角索引
):
    """
    主推理函数：加载Rig COLMAP采样数据并运行DepthAnything3推理
    支持Rig格式的COLMAP数据（包含rigs.bin和frames.bin）
    支持根据硬编码的图片索引和相机索引列表进行视角选择
    
    Args:
        colmap_dir: COLMAP根目录（包含images/和sparse/子目录）
        skip_step: 图片采样步长
        model_name: 预训练模型名称/路径
        export_dir: 结果导出目录
        export_format: 导出格式（支持组合：mini_npz/glb/gs_ply等）
        device: 运行设备（cuda/cpu）
        image_indices: 图片索引列表（排序后的第几张，从0开始），None表示使用所有图片
        camera_indices: 相机索引列表（pano_camera编号），None表示使用所有相机
    """
    # 1. 初始化模型
    model = DepthAnything3.from_pretrained(model_name).to(device)
    model.eval()
    
    # 2. 加载采样后的Rig COLMAP数据
    image_paths, extrinsics, intrinsics = load_rig_colmap_data_with_skip(
        colmap_dir=colmap_dir,
        sparse_subdir="0",  # 根据实际COLMAP目录调整
        skip_step=skip_step,
        image_indices=image_indices,
        camera_indices=camera_indices
    )
    # 导出配置
    export_kwargs = {
        "gs_video": {
            "trj_mode": "original",  # 使用原始相机轨迹
            "video_quality": "high", # 可选：低/中/高，对应 low/medium/high
        },
    }
    # 如果提供了 pandepth_indices，则传给 gs_pandepth 导出器
    if pandepth_indices is not None:
        export_kwargs["gs_pandepth"] = {
            "pandepth_indices": pandepth_indices,
        }
    # 3. 运行推理（带姿态条件的深度估计）
    prediction = model.inference(
        # 输入数据（采样后的图片+相机参数）
        image=image_paths,
        extrinsics=extrinsics,  # COLMAP外参 (N,4,4)
        intrinsics=intrinsics,  # COLMAP内参 (N,3,3)
        
        # 核心参数
        align_to_input_ext_scale=False,  # 对齐输入外参尺度
        use_ray_pose=True,  # 可选：启用ray-based pose提升精度（稍慢）
        ref_view_strategy="saddle_balanced",  # 多视角参考帧选择策略
        infer_gs=True,
        
        # 处理分辨率
        process_res=504,
        process_res_method="upper_bound_resize",
        
        # 导出配置
        export_dir=export_dir,
        export_format=export_format,
        conf_thresh_percentile=40.0,  # GLB导出置信度阈值
        num_max_points=30_000_000,     # GLB点云最大点数
        show_cameras=True,            # GLB中显示相机位姿
        export_kwargs=export_kwargs,
    )
    
    # 4. 输出结果信息
    print(f"\n推理完成！结果导出至: {export_dir}")
    print(f"  深度图形状: {prediction.depth.shape}")
    print(f"  相机外参形状: {prediction.extrinsics.shape}")
    print(f"  相机内参形状: {prediction.intrinsics.shape}")
    
    return prediction

# -------------------------- 示例调用 --------------------------
if __name__ == "__main__":
    # 示例1：使用所有图片（skip_step=1）
    # run_rig_colmap_inference_with_skip(
    #     colmap_dir="./path/to/rig_colmap_data",
    #     skip_step=1,
    #     export_dir="./output_all"
    # )
    
    # 示例2：从innovation32 rig colmap数据加载，每25张图片取1张（skip_step=25）
    # 硬编码的视角选择列表（参考extract_images.py）
    # 列表A：图片索引（排序后的第几张，从0开始）
    A = [0,2,4]  # 示例：提取第0、1、2、3张图片(等价为第i个全景图)
    
    # 列表B：相机索引（pano_camera编号）
    B = [12, 15, 19, 22]  # 示例：从这些相机目录提取

    # 列表C：用于全景深度导出的视角索引（等价于 Prediction 中的视图索引）
    # 这些索引应对应于推理结果中的视图顺序（即 depth / extrinsics / intrinsics 的第 Ci 帧）。
    C = [0,1,2,15]  # 示例：从这些视图位置生成全景深度 cubemap

    # 为 gs_pandepth 导出方式增加配置
    # 注意：这里直接修改 export_format，增加 "gs_pandepth" 选项
    # export_format = "mini_npz-glb-gs_ply-gs_video-gs_pandepth"
    export_format = "gs_ply-gs_pandepth"

    run_rig_colmap_inference_with_skip(
        colmap_dir="/root/autodl-tmp/data/colmap_360Roam_4x/bar16",
        skip_step=1,
        export_dir=f"/root/autodl-tmp/results/rigcolmap/bar16_{len(A)}p{len(B)}v",
        export_format=export_format,
        image_indices=A,  # 使用硬编码的图片索引列表
        camera_indices=B,  # 使用硬编码的相机索引列表
        pandepth_indices=C,  # 将列表 C 传给 gs_pandepth
    )