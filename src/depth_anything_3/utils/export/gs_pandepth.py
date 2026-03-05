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


def _cubemap_to_equirect(
    color_all: torch.Tensor,  # (6, 3, H, W), float32 in [0, 1]
    depth_all: torch.Tensor,  # (6, H, W), float32
    intrinsics: torch.Tensor,  # (3, 3)
    rel_rots: torch.Tensor,  # (6, 3, 3)
    pano_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 6 个 cubemap 视角的 RGB + 深度“像素级重采样”为一个等轴柱状全景图（颜色 + 深度）。

    不再做 3D 点云重投影和深度筛选，而是：
      1. 为每个全景像素生成一个方向向量；
      2. 用标准 cubemap 规则判定该方向落在哪个面；
      3. 在对应的 cubemap 面上采样颜色和深度。

    返回：
      pano_depth: (Hp, Wp) float32
      pano_color: (Hp, Wp, 3) uint8
    """
    assert color_all.shape[0] == depth_all.shape[0] == 6

    device = color_all.device
    _, _, Hf, Wf = color_all.shape

    if pano_height is None:
        pano_height = Hf
    pano_width = pano_height * 2

    # 1. 构建等轴柱状全景的方向向量
    yy, xx = torch.meshgrid(
        torch.arange(pano_height, dtype=torch.float32, device=device),
        torch.arange(pano_width, dtype=torch.float32, device=device),
        indexing="ij",
    )  # (Hp, Wp)

    u = (xx + 0.5) / pano_width   # [0,1]
    v = (yy + 0.5) / pano_height  # [0,1]

    yaw = (2.0 * u - 1.0) * np.pi           # [-pi, pi]
    pitch = (1.0 - 2.0 * v) * (np.pi / 2.0)  # [-pi/2, pi/2]

    dx = torch.sin(yaw) * torch.cos(pitch)
    # 注意：这里使用 dy = +sin(pitch)，保持与渲染 cubemap 时的相机坐标系一致，
    # 避免上下极区在融合时出现“翻转”错位。
    dy = torch.sin(pitch)
    dz = torch.cos(yaw) * torch.cos(pitch)

    # 2. 按最大分量决定 cubemap 面（与 _build_cubemap_relative_rotations 的立方体方向一致）
    ax = dx.abs()
    ay = dy.abs()
    az = dz.abs()

    face_idx = torch.zeros_like(ax, dtype=torch.long)

    # X 轴主导：+X 右侧面 (1)，-X 左侧面 (3)
    is_x_major = (ax >= ay) & (ax >= az)
    face_idx[is_x_major & (dx > 0)] = 1  # +X -> 右
    face_idx[is_x_major & (dx < 0)] = 3  # -X -> 左

    # Y 轴主导：+Y 上侧面 (4)，-Y 下侧面 (5)
    is_y_major = (ay > ax) & (ay >= az)
    face_idx[is_y_major & (dy > 0)] = 4  # +Y -> 上
    face_idx[is_y_major & (dy < 0)] = 5  # -Y -> 下

    # 其余为 Z 轴主导：+Z 前面 (0)，-Z 后面 (2)
    is_z_major = ~(is_x_major | is_y_major)
    face_idx[is_z_major & (dz > 0)] = 0  # +Z -> 前
    face_idx[is_z_major & (dz < 0)] = 2  # -Z -> 后

    # 3. 计算每个像素在对应 cubemap 面上的局部坐标 (u_face, v_face) ∈ [-1,1]
    u_face = torch.zeros_like(dx)
    v_face = torch.zeros_like(dy)

    # +X
    mask = face_idx == 1
    ax_safe = ax[mask].clamp_min(1e-6)
    u_face[mask] = -dz[mask] / ax_safe
    v_face[mask] = -dy[mask] / ax_safe

    # -X
    mask = face_idx == 3
    ax_safe = ax[mask].clamp_min(1e-6)
    u_face[mask] = dz[mask] / ax_safe
    v_face[mask] = -dy[mask] / ax_safe

    # +Y
    mask = face_idx == 4
    ay_safe = ay[mask].clamp_min(1e-6)
    u_face[mask] = dx[mask] / ay_safe
    v_face[mask] = dz[mask] / ay_safe

    # -Y
    mask = face_idx == 5
    ay_safe = ay[mask].clamp_min(1e-6)
    u_face[mask] = dx[mask] / ay_safe
    v_face[mask] = -dz[mask] / ay_safe

    # +Z
    mask = face_idx == 0
    az_safe = az[mask].clamp_min(1e-6)
    u_face[mask] = dx[mask] / az_safe
    v_face[mask] = -dy[mask] / az_safe

    # -Z
    mask = face_idx == 2
    az_safe = az[mask].clamp_min(1e-6)
    u_face[mask] = -dx[mask] / az_safe
    v_face[mask] = -dy[mask] / az_safe

    # 4. 将 [-1,1] 的 (u_face, v_face) 转成 cubemap 像素坐标，并从对应面采样
    pano_color = torch.zeros(
        (3, pano_height, pano_width), dtype=color_all.dtype, device=device
    )
    pano_depth = torch.zeros(
        (pano_height, pano_width), dtype=depth_all.dtype, device=device
    )

    # 归一化到 [0, Wf-1] / [0, Hf-1]
    x_tex = ((u_face + 1.0) * 0.5 * (Wf - 1)).clamp(0, Wf - 1)
    y_tex = ((v_face + 1.0) * 0.5 * (Hf - 1)).clamp(0, Hf - 1)

    x_tex_long = x_tex.round().long()
    y_tex_long = y_tex.round().long()

    for k in range(6):
        mask_k = face_idx == k
        if not mask_k.any():
            continue

        ys_k, xs_k = torch.nonzero(mask_k, as_tuple=True)
        xs_src = x_tex_long[mask_k]
        ys_src = y_tex_long[mask_k]

        # 水平方向 0,1,2,3 与 cubemap 保持一致；
        # 仅在融合时将 cubemap 的上/下两个面对调：
        #   equirect 判定到面 4(上) -> 实际去采样 cubemap 的面 5；
        #   equirect 判定到面 5(下) -> 实际去采样 cubemap 的面 4。
        src_k = k
        if k == 4:
            src_k = 5
        elif k == 5:
            src_k = 4

        # 颜色：3×H×W
        src_color = color_all[src_k, :, ys_src, xs_src]  # (3, N)
        pano_color[:, ys_k, xs_k] = src_color

        # 深度：H×W
        src_depth = depth_all[src_k, ys_src, xs_src]  # (N,)
        pano_depth[ys_k, xs_k] = src_depth

    pano_color_np = (
        pano_color.clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )  # (Hp, Wp, 3)
    pano_depth_np = pano_depth.cpu().numpy()  # (Hp, Wp)

    return pano_depth_np, pano_color_np


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

        # -----------------------
        # 1）保留原来的 cubemap 2×6 网格图：point[ci].jpg
        # -----------------------
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

        row_color = hcat(*color_tiles, align="center", gap=4, gap_color=0)
        row_depth = hcat(*depth_tiles, align="center", gap=4, gap_color=0)
        combined_grid = vcat(row_color, row_depth, align="center", gap=4, gap_color=0)

        # 可选：降低分辨率再保存，这里统一缩小到原来的 0.5 倍
        DOWNSAMPLE_RATIO = 0.5
        if DOWNSAMPLE_RATIO is not None and DOWNSAMPLE_RATIO < 1.0:
            combined_grid_to_save = F.interpolate(
                combined_grid.unsqueeze(0),
                scale_factor=DOWNSAMPLE_RATIO,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            combined_grid_to_save = combined_grid

        combined_grid_np = (
            combined_grid_to_save.clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        save_path_grid = os.path.join(out_dir, f"point[{ci}].jpg")
        imageio.imwrite(save_path_grid, combined_grid_np, quality=95)

        # -----------------------
        # 2）基于 6 个 cubemap 视角融合为等轴柱状全景：point[ci]_pan.jpg
        # -----------------------
        pano_depth_np, pano_color_np = _cubemap_to_equirect(
            color_all=color_all,
            depth_all=depth_all,
            intrinsics=intr_ci,
            rel_rots=rel_rots,
            pano_height=H,
        )

        pano_color_t = (
            torch.from_numpy(pano_color_np)
            .to(device=device)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )  # (3, Hp, Wp)
        pano_depth_t = torch.from_numpy(pano_depth_np).to(device=device)  # (Hp, Wp)
        pano_depth_vis_t = vis_depth_map_tensor(pano_depth_t).clamp(0.0, 1.0)

        combined_pan = vcat(
            pano_color_t, pano_depth_vis_t, align="center", gap=4, gap_color=0
        )

        if DOWNSAMPLE_RATIO is not None and DOWNSAMPLE_RATIO < 1.0:
            combined_pan_to_save = F.interpolate(
                combined_pan.unsqueeze(0),
                scale_factor=DOWNSAMPLE_RATIO,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            combined_pan_to_save = combined_pan

        combined_pan_np = (
            combined_pan_to_save.clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        save_path_pan = os.path.join(out_dir, f"point[{ci}]_pan.jpg")
        imageio.imwrite(save_path_pan, combined_pan_np, quality=95)

    print(f"Exported {len(pandepth_indices)} gs_pandepth images to {out_dir}")

