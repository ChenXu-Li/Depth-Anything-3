# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np
import torch

from depth_anything_3.specs import Prediction
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.utils.visualize import visualize_depth, vis_depth_map_tensor
from depth_anything_3.utils.layout_helpers import hcat, vcat


def export_to_gs_depth(
    prediction: Prediction,
    export_dir: str,
    **kwargs,
):
    """
    为每个输入视图生成合并图像，包含：
    - 原图 (Pi_gt)
    - 高斯渲染图 (Pi_gs)
    - 预测深度图 (Pi_d)
    - 高斯渲染深度图 (Pi_gsd)
    
    四张图以2x2布局合并保存。
    
    Args:
        prediction: 推理结果
        export_dir: 导出目录
    """
    if prediction.gaussians is None:
        raise ValueError("prediction.gaussians is required but not available")
    
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")
    
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError("prediction.extrinsics and intrinsics are required but not available")
    
    os.makedirs(os.path.join(export_dir, "gs_depth"), exist_ok=True)
    
    # 获取数据
    gaussians = prediction.gaussians
    processed_images = prediction.processed_images  # (N, H, W, 3) uint8
    depth_pred = prediction.depth  # (N, H, W)
    extrinsics = torch.from_numpy(prediction.extrinsics).to(gaussians.means)  # (N, 4, 4) or (N, 3, 4)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(gaussians.means)  # (N, 3, 3)
    
    N = depth_pred.shape[0]
    H, W = depth_pred.shape[1], depth_pred.shape[2]
    
    # Ensure extrinsics are homogeneous 4x4 matrices (world-to-camera).
    # Some pipelines store COLMAP-style extrinsics as 3x4 [R|t].
    if extrinsics.shape[-2:] == (3, 4):
        extrinsics_h = torch.zeros(
            (extrinsics.shape[0], 4, 4), dtype=extrinsics.dtype, device=extrinsics.device
        )
        extrinsics_h[:, :3, :4] = extrinsics
        extrinsics_h[:, 3, 3] = 1.0
        extrinsics = extrinsics_h
    elif extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            f"Unsupported extrinsics shape: {tuple(extrinsics.shape)} (expected (N,4,4) or (N,3,4))"
        )

    # 处理尺度因子（如果存在）
    if prediction.is_metric and prediction.scale_factor is not None:
        scale_factor = prediction.scale_factor
        extrinsics_scaled = extrinsics.clone()
        extrinsics_scaled[:, :3, 3] /= scale_factor
    else:
        extrinsics_scaled = extrinsics
    
    # 使用与 `gs_video` 相同的渲染管线，在原始视角下渲染所有视图
    tgt_extrs = extrinsics_scaled.unsqueeze(0)  # (1, N, 4, 4)
    tgt_intrs = intrinsics.unsqueeze(0)  # (1, N, 3, 3)

    color_all, depth_all = run_renderer_in_chunk_w_trj_mode(
        gaussians=gaussians,
        extrinsics=tgt_extrs,          # world2cam, "b v 4 4"
        intrinsics=tgt_intrs,          # unnormed intrinsics, "b v 3 3"
        image_shape=(H, W),
        chunk_size=4,
        trj_mode="original",           # 使用原始视角轨迹
        color_mode="RGB+D",
        enable_tqdm=False,
    )

    # color_all: (1, N, 3, H, W); depth_all: (1, N, H, W)
    color_all = color_all[0]
    depth_all = depth_all[0]

    # 为每个视图生成合并图像
    for i in range(N):
        # 1. 原图 (Pi_gt)
        img_gt = processed_images[i]  # (H, W, 3) uint8
        
        # 2. 高斯渲染图 (Pi_gs) 和深度图 (Pi_gsd)，使用与该视图对应的渲染结果
        img_gs_tensor = color_all[i]      # (3, H, W)
        depth_gs = depth_all[i]           # (H, W)

        img_gs = (img_gs_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
        
        # 3. 预测深度图 (Pi_d)
        depth_pred_vis = visualize_depth(depth_pred[i])  # (H, W, 3) uint8
        
        # 4. 高斯渲染深度图 (Pi_gsd)
        depth_gs_vis_tensor = vis_depth_map_tensor(depth_gs)  # (3, H, W) float [0, 1]
        depth_gs_vis = (depth_gs_vis_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
        
        # 转换为torch tensor用于布局合并 (C, H, W) float [0, 1]
        img_gt_t = torch.from_numpy(img_gt).permute(2, 0, 1).float() / 255.0
        img_gs_t = torch.from_numpy(img_gs).permute(2, 0, 1).float() / 255.0
        depth_pred_vis_t = torch.from_numpy(depth_pred_vis).permute(2, 0, 1).float() / 255.0
        depth_gs_vis_t = torch.from_numpy(depth_gs_vis).permute(2, 0, 1).float() / 255.0
        
        # 合并为2x2布局
        # 第一行：原图 | 高斯渲染图
        top_row = hcat(img_gt_t, img_gs_t, align="center", gap=4, gap_color=0)
        # 第二行：预测深度图 | 高斯渲染深度图
        bottom_row = hcat(depth_pred_vis_t, depth_gs_vis_t, align="center", gap=4, gap_color=0)
        # 垂直合并
        combined = vcat(top_row, bottom_row, align="center", gap=4, gap_color=0)
        
        # 转换回numpy uint8并保存
        combined_np = (combined.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H', W', 3) uint8
        
        save_path = os.path.join(export_dir, f"gs_depth/{i:04d}.jpg")
        imageio.imwrite(save_path, combined_np, quality=95)
    
    print(f"Exported {N} gs_depth comparison images to {export_dir}/gs_depth/")
