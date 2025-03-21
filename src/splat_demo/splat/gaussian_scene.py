import time
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from tqdm import tqdm

from splat.gaussians import Gaussians
from splat.image import GaussianImage
from splat.schema import PreprocessedScene
from splat.utils import (
    compute_2d_covariance,
    compute_extent_and_radius,
    compute_gaussian_weight,
    compute_inverted_covariance,
    in_view_frustum,
    load_cuda,
    ndc2Pix,
    read_camera_file,
    read_image_file,
)


class GaussianScene(nn.Module):
    def __init__(
        self,
        colmap_path: str,
        gaussians: Gaussians,
    ) -> None:
        super().__init__()

        camera_dict = read_camera_file(colmap_path)
        image_dict = read_image_file(colmap_path)
        self.images = {}
        for idx in image_dict.keys():
            image = image_dict[idx]
            camera = camera_dict[image.camera_id]
            image = GaussianImage(camera=camera, image=image)
            self.images[idx] = image

        self.gaussians = gaussians

    def render_points_image(self, image_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function that can be gotten rid of once we know
        implementation is correct
        """
        return self.images[image_idx].project_point_to_camera_perspective_projection(
            self.gaussians.points, self.gaussians.colors
        )

    def get_2d_covariance(
        self, image_idx: int, points: torch.Tensor, covariance_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the 2D covariance matrix for each gaussian
        """
        output = compute_2d_covariance(
            points=points,
            extrinsic_matrix=self.images[image_idx].world2view.to(points.device),
            covariance_3d=covariance_3d,
            tan_fovX=self.images[image_idx].tan_fovX.to(points.device),
            tan_fovY=self.images[image_idx].tan_fovY.to(points.device),
            focal_x=self.images[image_idx].f_x.to(points.device),
            focal_y=self.images[image_idx].f_y.to(points.device),
        )
        return output

    def preprocess(self, image_idx: int) -> None:
        """Preprocesses before rendering begins"""
        in_view = in_view_frustum(
            points=self.gaussians.points,
            view_matrix=self.images[image_idx].world2view,
        )
        covariance_3d = self.gaussians.get_3d_covariance_matrix()[in_view]

        points = self.gaussians.points[in_view]
        points_homogeneous = torch.cat(
            [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
        )
        points_view = (
            points_homogeneous
            @ self.images[image_idx].world2view.to(points_homogeneous.device)
        )[:, :3]

        points_ndc = points_homogeneous @ self.images[image_idx].full_proj_transform.to(
            points_homogeneous.device
        )
        points_ndc = points_ndc[:, :3] / points_ndc[:, 3].unsqueeze(1)
        points_xy = points_ndc[:, :2]
        points_xy[:, 0] = ndc2Pix(
            points_xy[:, 0], self.images[image_idx].width.to(points_xy.device)
        )
        points_xy[:, 1] = ndc2Pix(
            points_xy[:, 1], self.images[image_idx].height.to(points_xy.device)
        )

        covariance_2d = self.get_2d_covariance(
            image_idx=image_idx, points=points, covariance_3d=covariance_3d
        )

        inverse_covariance = compute_inverted_covariance(covariance_2d)
        # now we compute the radius
        # radius = self.compute_radius(covariance_2d, determinant)
        radius = compute_extent_and_radius(covariance_2d)

        min_x = torch.floor(points_xy[:, 0] - radius)
        min_y = torch.floor(points_xy[:, 1] - radius)
        max_x = torch.ceil(points_xy[:, 0] + radius)
        max_y = torch.ceil(points_xy[:, 1] + radius)

        # sort by depth
        colors = self.gaussians.colors[in_view]
        opacity = self.gaussians.opacity[in_view]

        indices_by_depth = torch.argsort(points_view[:, 2])
        points_view = points_view[indices_by_depth]
        colors = colors[indices_by_depth]
        opacity = opacity[indices_by_depth]
        points = points_xy[indices_by_depth]
        covariance_2d = covariance_2d[indices_by_depth]
        inverse_covariance = inverse_covariance[indices_by_depth]
        radius = radius[indices_by_depth]
        points_xy = points_xy[indices_by_depth]
        min_x = min_x[indices_by_depth]
        min_y = min_y[indices_by_depth]
        max_x = max_x[indices_by_depth]
        max_y = max_y[indices_by_depth]

        return PreprocessedScene(
            points=points,
            colors=colors,
            covariance_2d=covariance_2d,
            depths=points_view[:, 2],
            inverse_covariance_2d=inverse_covariance,
            radius=radius,
            points_xy=points_xy,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            sigmoid_opacity=torch.sigmoid(opacity),
        )

    def render_pixel(
        self,
        pixel_coords: torch.Tensor,
        points_in_tile_mean: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        inverse_covariance: torch.Tensor,
        min_weight: float = 0.000001,
    ) -> torch.Tensor:
        total_weight = torch.ones(1).to(points_in_tile_mean.device)
        pixel_color = torch.zeros((1, 1, 3)).to(points_in_tile_mean.device)
        for point_idx in range(points_in_tile_mean.shape[0]):
            point = points_in_tile_mean[point_idx, :].view(1, 2)
            weight = compute_gaussian_weight(
                pixel_coord=pixel_coords,
                point_mean=point,
                inverse_covariance=inverse_covariance[point_idx],
            )
            alpha = weight * torch.sigmoid(opacities[point_idx])
            test_weight = total_weight * (1 - alpha)
            if test_weight < min_weight:
                return pixel_color
            pixel_color += total_weight * alpha * colors[point_idx]
            total_weight = test_weight
        # in case we never reach saturation
        return pixel_color

    def render_tile(
        self,
        x_min: int,
        y_min: int,
        points_in_tile_mean: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        inverse_covariance: torch.Tensor,
        tile_size: int = 16,
    ) -> torch.Tensor:
        """Points in tile should be arranged in order of depth"""

        tile = torch.zeros((tile_size, tile_size, 3))

        for pixel_x in range(x_min, x_min + tile_size):
            for pixel_y in range(y_min, y_min + tile_size):
                tile[pixel_x % tile_size, pixel_y % tile_size] = self.render_pixel(
                    pixel_coords=torch.Tensor([pixel_x, pixel_y])
                    .view(1, 2)
                    .to(points_in_tile_mean.device),
                    points_in_tile_mean=points_in_tile_mean,
                    colors=colors,
                    opacities=opacities,
                    inverse_covariance=inverse_covariance,
                )
        return tile

    def render_image(self, image_idx: int, tile_size: int = 16) -> torch.Tensor:
        """For each tile have to check if the point is in the tile"""
        preprocessed_scene = self.preprocess(image_idx)
        height = int(self.images[image_idx].height.item())
        width = int(self.images[image_idx].width.item())

        image = torch.zeros((width, height, 3))

        for x_min in tqdm(range(0, width - tile_size, tile_size)):
            x_in_tile = (preprocessed_scene.min_x <= x_min + tile_size) & (
                preprocessed_scene.max_x >= x_min
            )
            if x_in_tile.sum() == 0:
                continue
            for y_min in range(0, height - tile_size, tile_size):
                y_in_tile = (preprocessed_scene.min_y <= y_min + tile_size) & (
                    preprocessed_scene.max_y >= y_min
                )
                points_in_tile = x_in_tile & y_in_tile
                if points_in_tile.sum() == 0:
                    continue
                points_in_tile_mean = preprocessed_scene.points[points_in_tile]
                colors_in_tile = preprocessed_scene.colors[points_in_tile]
                opacities_in_tile = preprocessed_scene.sigmoid_opacity[points_in_tile]
                inverse_covariance_in_tile = preprocessed_scene.inverse_covariance_2d[
                    points_in_tile
                ]
                image[x_min : x_min + tile_size, y_min : y_min + tile_size] = (
                    self.render_tile(
                        x_min=x_min,
                        y_min=y_min,
                        points_in_tile_mean=points_in_tile_mean,
                        colors=colors_in_tile,
                        opacities=opacities_in_tile,
                        inverse_covariance=inverse_covariance_in_tile,
                        tile_size=tile_size,
                    )
                )
        return image

    def compile_cuda_ext(
        self, src_path = "./"
    ) -> torch.jit.ScriptModule:

        cpp_src = """
        torch::Tensor render_image(
            int image_height,
            int image_width,
            int tile_size,
            torch::Tensor point_means,
            torch::Tensor point_colors,
            torch::Tensor inverse_covariance_2d,
            torch::Tensor min_x,
            torch::Tensor max_x,
            torch::Tensor min_y,
            torch::Tensor max_y,
            torch::Tensor opacity);
        """

        cuda_src = Path(f"{src_path}/splat/c/render.cu").read_text()

        return load_cuda(cuda_src, cpp_src, ["render_image"], opt=True, verbose=True)

    def render_image_cuda(self, image_idx: int, tile_size: int = 16, src_path: str = "") -> torch.Tensor:
        preprocessed_scene = self.preprocess(image_idx)
        height = self.images[image_idx].height
        width = self.images[image_idx].width
        ext = self.compile_cuda_ext(src_path)

        now = time.time()
        image = ext.render_image(
            height,
            width,
            tile_size,
            preprocessed_scene.points.contiguous(),
            preprocessed_scene.colors.contiguous(),
            preprocessed_scene.inverse_covariance_2d.contiguous(),
            preprocessed_scene.min_x.contiguous(),
            preprocessed_scene.max_x.contiguous(),
            preprocessed_scene.min_y.contiguous(),
            preprocessed_scene.max_y.contiguous(),
            preprocessed_scene.sigmoid_opacity.contiguous(),
        )
        torch.cuda.synchronize()
        print("Operation took seconds: ", time.time() - now)
        return image
