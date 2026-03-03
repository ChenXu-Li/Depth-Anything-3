import os
from typing import Iterable, List, Sequence

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.layout_helpers import hcat, vcat
from depth_anything_3.utils.visualize import visualize_depth, vis_depth_map_tensor


def _ensure_homogeneous_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Ensure extrinsics are 4x4 homogeneous world-to-camera matrices.
    Accepts shapes (N, 4, 4) or (N, 3, 4).
    """
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        extrinsics_h = torch.zeros(
            (extrinsics.shape[0], 4, 4), dtype=extrinsics.dtype, device=extrinsics.device
        )
        extrinsics_h[:, :3, :4] = extrinsics
        extrinsics_h[:, 3, 3] = 1.0
        return extrinsics_h
    raise ValueError(
        f"Unsupported extrinsics shape: {tuple(extrinsics.shape)} (expected (N,4,4) or (N,3,4))"
    )


def _build_cubemap_relative_rotations(device: torch.device) -> torch.Tensor:
    """
    Build 6 relative rotations in the original camera coordinate system for cubemap faces.

    The canonical camera looks along +Z with +X to the right and +Y up.
    For each cubemap face we define (right, up, forward) in this canonical space.
    The resulting 3x3 matrices map new-camera coordinates to original-camera coordinates.
    """
    # Definitions follow a common cubemap convention.
    # Each matrix has columns (right, up, forward) in original camera coords.
    # 这里按照用户指定的顺序排列 6 个面：
    #   0: 中心（原始 pano_camera12 方向，对应 front/+Z）
    #   1: 右 (+X)
    #   2: 后 (-Z)
    #   3: 左 (-X)
    #   4: 上 (+Y)
    #   5: 下 (-Y)
    faces: List[torch.Tensor] = []

    def make_R(right, up, forward):
        # Stack as columns: [right, up, forward]
        return torch.stack(
            [
                torch.tensor(right, dtype=torch.float32, device=device),
                torch.tensor(up, dtype=torch.float32, device=device),
                torch.tensor(forward, dtype=torch.float32, device=device),
            ],
            dim=-1,
        )

    # 0: front / original forward (+Z)  -> 中
    faces.append(make_R((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))
    # 1: right (+X) -> 右
    faces.append(make_R((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)))
    # 2: back (-Z) -> 后
    faces.append(make_R((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    # 3: left (-X) -> 左
    faces.append(make_R((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)))
    # 4: top (+Y) -> 上
    faces.append(make_R((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    # 5: bottom (-Y) -> 下
    faces.append(make_R((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)))

    return torch.stack(faces, dim=0)  # (6, 3, 3)


def export_to_gs_pandepth(
    prediction: Prediction,
    export_dir: str,
    pandepth_indices: Sequence[int] | None = None,
) -> None:
    """
    Export panoramic depth comparison images for selected viewpoints.

    For each index Ci in `pandepth_indices`, we:
      - Take the corresponding camera pose as the origin.
      - Use that pose to define the local camera coordinate frame.
      - Construct 6 cubemap directions around this origin.
      - Render, for each of the 6 directions:
          * Gaussian RGB rendering
          * Gaussian depth map (visualized)
          * Predicted depth map (from the original view Ci, shared across the 6 rows)
      - Arrange them into a 6 (rows) x 3 (cols) grid and save as:
            gs_pandepth/point[Ci].jpg

    Args:
        prediction: Inference prediction containing gaussians, depth, extrinsics, intrinsics.
        export_dir: Root export directory.
        pandepth_indices: Sequence of integer view indices Ci. If None, defaults to all views.
                          Each Ci is interpreted as an index into prediction.{depth,extrinsics,intrinsics}.
    """
    if prediction.gaussians is None:
        raise ValueError("prediction.gaussians is required but not available")
    if prediction.depth is None:
        raise ValueError("prediction.depth is required but not available")
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError("prediction.extrinsics and intrinsics are required but not available")

    gaussians = prediction.gaussians
    depth_pred_np: np.ndarray = prediction.depth  # (N, H, W)
    extrinsics_np: np.ndarray = prediction.extrinsics  # (N, 4, 4) or (N, 3, 4)
    intrinsics_np: np.ndarray = prediction.intrinsics  # (N, 3, 3)

    N, H, W = depth_pred_np.shape

    device = gaussians.means.device
    extrinsics = _ensure_homogeneous_extrinsics(
        torch.from_numpy(extrinsics_np).to(device=device, dtype=gaussians.means.dtype)
    )  # (N, 4, 4)
    intrinsics = torch.from_numpy(intrinsics_np).to(device=device, dtype=gaussians.means.dtype)

    # Handle metric scaling if available (consistent with other exporters).
    if getattr(prediction, "is_metric", 0) and prediction.scale_factor is not None:
        scale_factor = prediction.scale_factor
        extrinsics_scaled = extrinsics.clone()
        extrinsics_scaled[:, :3, 3] /= scale_factor
    else:
        extrinsics_scaled = extrinsics

    # Determine which indices to process.
    if pandepth_indices is None:
        pandepth_indices = list(range(N))
    else:
        pandepth_indices = list(pandepth_indices)

    # Prepare output directory.
    out_dir = os.path.join(export_dir, "gs_pandepth")
    os.makedirs(out_dir, exist_ok=True)

    # Precompute relative cubemap rotations in camera coordinates.
    rel_rots = _build_cubemap_relative_rotations(device=device)  # (6, 3, 3)

    for ci in pandepth_indices:
        if ci < 0 or ci >= N:
            raise IndexError(
                f"pandepth index {ci} is out of range for number of views {N}"
            )

        # Base pose: world-to-camera (4x4)
        base_w2c = extrinsics_scaled[ci]  # (4, 4)

        # Convert to camera-to-world to get origin and orientation.
        base_c2w = torch.inverse(base_w2c)  # (4, 4)
        R_c2w = base_c2w[:3, :3]  # (3, 3)
        t_c2w = base_c2w[:3, 3]   # (3,)

        # Build 6 cubemap world-to-camera matrices around the same origin.
        cubemap_w2c: list[torch.Tensor] = []
        for r_rel in rel_rots:
            # New camera-to-world rotation: R_c2w_new = R_c2w @ R_cam_rel
            R_c2w_new = R_c2w @ r_rel  # (3, 3)
            # Convert back to world-to-camera.
            R_w2c_new = R_c2w_new.transpose(0, 1)
            t_w2c_new = -R_w2c_new @ t_c2w

            m = torch.eye(4, dtype=gaussians.means.dtype, device=device)
            m[:3, :3] = R_w2c_new
            m[:3, 3] = t_w2c_new
            cubemap_w2c.append(m)

        cubemap_w2c_t = torch.stack(cubemap_w2c, dim=0)  # (6, 4, 4)

        # Use intrinsics of the base view for all 6 faces.
        intr_ci = intrinsics[ci]  # (3, 3)
        intr_cubemap = intr_ci.unsqueeze(0).repeat(cubemap_w2c_t.shape[0], 1, 1)  # (6, 3, 3)

        # Run renderer: batch dimension = 1, views = 6.
        tgt_extrs = cubemap_w2c_t.unsqueeze(0)  # (1, 6, 4, 4)
        tgt_intrs = intr_cubemap.unsqueeze(0)   # (1, 6, 3, 3)

        color_all, depth_all = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=tgt_extrs,
            intrinsics=tgt_intrs,
            image_shape=(H, W),
            chunk_size=4,
            trj_mode="original",  # 直接使用提供的 cubemap 视角，不做轨迹插值，避免旋转矩阵奇异检查
            color_mode="RGB+D",
            enable_tqdm=False,
        )

        # color_all: (1, 6, 3, H, W); depth_all: (1, 6, H, W)
        color_all = color_all[0]
        depth_all = depth_all[0]

        # 翻转渲染结果的垂直方向，使之与训练/输入视图的坐标系一致（防止上下颠倒）
        color_all = torch.flip(color_all, dims=[-2])  # flip H
        depth_all = torch.flip(depth_all, dims=[-2])  # flip H

        # Build 2 rows × 6 columns：
        #   第 1 行：6 个方向的高斯渲染图（按：中、右、后、左、上、下）
        #   第 2 行：对应 6 个方向的高斯深度图
        color_tiles: list[torch.Tensor] = []
        depth_tiles: list[torch.Tensor] = []
        for face_idx in range(color_all.shape[0]):  # 按 0..5 的固定顺序
            # Gaussian RGB
            img_gs_tensor = color_all[face_idx]  # (3, H, W), float in [0,1]
            img_gs_t = img_gs_tensor.clamp(0.0, 1.0)

            # Gaussian depth (visualized)
            depth_gs = depth_all[face_idx]  # (H, W)
            depth_gs_vis_tensor = vis_depth_map_tensor(depth_gs)  # (3, H, W) float [0,1]
            depth_gs_vis_t = depth_gs_vis_tensor.clamp(0.0, 1.0)

            color_tiles.append(img_gs_t)
            depth_tiles.append(depth_gs_vis_t)

        # 水平拼接得到两行：2×6 网格
        row_color = hcat(*color_tiles, align="center", gap=4, gap_color=0)
        row_depth = hcat(*depth_tiles, align="center", gap=4, gap_color=0)
        combined = vcat(row_color, row_depth, align="center", gap=4, gap_color=0)

        # 可选：降低分辨率再保存，这里统一缩小到原来的 0.5 倍
        DOWNSAMPLE_RATIO = 0.5
        if DOWNSAMPLE_RATIO is not None and DOWNSAMPLE_RATIO < 1.0:
            combined = F.interpolate(
                combined.unsqueeze(0),
                scale_factor=DOWNSAMPLE_RATIO,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Convert to uint8 image and save.
        combined_np = (
            combined.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
        )  # (H_total, W_total, 3) uint8

        save_path = os.path.join(out_dir, f"point[{ci}].jpg")
        imageio.imwrite(save_path, combined_np, quality=95)

    print(f"Exported {len(pandepth_indices)} gs_pandepth images to {out_dir}")

