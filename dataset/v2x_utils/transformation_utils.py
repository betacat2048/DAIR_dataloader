from typing import Callable

import numpy as np
import pathlib
import json

from Transform3D import Quaternion, Transform, CoordinateSystem


# (..., 3), (..., 7) ==> (..., 8, 3)
def get_3d_8points(obj_dim: np.ndarray, pose: Transform) -> np.ndarray:
    pts = np.array(
        [
            [-1, +1, -1, +1, -1, +1, -1, +1],
            [-1, -1, +1, +1, -1, -1, +1, +1],
            [0, 0, 0, 0, 2, 2, 2, 2],
        ]
    ).T / 2
    return Transform(pose.p[..., None, :])(pts * obj_dim[..., None, :])


# (N, 8, 3), (3, 3) ==> (N, 8, 2), (N, 8)
def project_points_to_image(pts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uvw = np.einsum('ij, ...j -> ...i', K, pts_cam)
    uv = uvw[..., :2] / uvw[..., 2:3]
    return uv, pts_cam[..., 2]


def draw_bboxes_points_and_wireframe(
        pts_cam: np.ndarray,
        K: np.ndarray,
        ax,
        w: int, h: int,
        point_mark: str = 'r.',  # same as your original
        line_width: float = 1.0,
        bound_r: float = 0.2,
):
    uv, z = project_points_to_image(pts_cam, K)

    in_front = (z > 0)  # (N, 8)

    u, v = uv[..., 0], uv[..., 1]
    in_bounds = (
            (-bound_r * w < u) & (u < (1 + bound_r) * w) &
            (-bound_r * h < v) & (v < (1 + bound_r) * h)
    )  # (N, 8)
    valid = np.mean((in_front & in_bounds).astype(float), axis=-1) > 0.0

    if valid.any():
        pts2d = uv[valid]  # (M, 8, 2)
        ax.plot(*pts2d.reshape(-1, 2).T, point_mark, markersize=2.0)

        EDGES = np.array([
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ], dtype=np.int32)

        segments = np.stack([pts2d[..., EDGES[:, 0], :], pts2d[..., EDGES[:, 1], :]], axis=-2)

        from matplotlib.collections import LineCollection
        lc = LineCollection(segments.reshape(-1, 2, 2), linewidths=line_width, colors='r')
        ax.add_collection(lc)


def polar_decompose_RS(M: np.ndarray, *, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    M = R @ S
    """
    # SVD: M = U Σ V^T
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T

    # Compute R = U V^T, ensure det(R) = +1 by flipping one axis if needed
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0  # Flip the last column of U
        s[-1] *= -1.0  # Keep M = U Σ V^T identity
        R = U @ Vt

    # Compute S = V Σ V^T, ensure non-negative singular values
    s_pos = np.maximum(s, tol)
    S = (V * s_pos) @ V.T  # Equivalent to V @ diag(s_pos) @ V^T

    return R, S


class CoordTransformation_xkc:
    def __init__(
            self,
            path_root: pathlib.Path,
            inf_frame,
            veh_frame,
    ):
        self.path_root = pathlib.Path(path_root)

        self.veh_frame = veh_frame
        self.inf_frame = inf_frame

        self.coord_system = CoordinateSystem('world')
        self.coord_system.add('world', 'veh_lidar',
                              self.load_transform_from_json(self.path_root / 'vehicle-side' / self.veh_frame['calib_novatel_to_world_path']) @
                              self.load_transform_from_json(self.path_root / 'vehicle-side' / self.veh_frame['calib_lidar_to_novatel_path'], key='transform')
                              )
        self.coord_system.add('world', 'inf_lidar', self.load_transform_from_json(self.path_root / 'infrastructure-side' / self.inf_frame['calib_virtuallidar_to_world_path'], apply_delta=True))

        self.coord_system.add('inf_lidar', 'inf', Transform.from_rot(Quaternion.from_euler(np.array(-90), np.array(0), np.array(0))))
        self.coord_system.add('veh_lidar', 'veh', Transform.from_rot(Quaternion.from_euler(np.array(-90), np.array(0), np.array(0))))

        inf_cam2lidar_trans, inf_cam2lidar_scale = self.load_transform_from_json(self.path_root / 'infrastructure-side' / self.inf_frame['calib_virtuallidar_to_camera_path'], allow_scale=True)
        self.inf_lidar2cam_scale = 1 / inf_cam2lidar_scale
        self.inf_lidar2cam_trans = inf_cam2lidar_trans.inverse()
        # self.coord_system.add('inf_lidar', 'inf_cam', inf_cam2lidar_trans.inverse())
        self.coord_system.add('veh_lidar', 'veh_cam', self.load_transform_from_json(self.path_root / 'vehicle-side' / self.veh_frame['calib_lidar_to_camera_path']).inverse())

    def inf_lidar2cam(self, x: np.ndarray) -> np.ndarray:
        return self.inf_lidar2cam_scale * self.inf_lidar2cam_trans(x)

    def inf_cam2lidar(self, x: np.ndarray) -> np.ndarray:
        return self.inf_lidar2cam_trans.inverse()(x / self.inf_lidar2cam_scale)

    @staticmethod
    def _to_transform(rot_matrix: np.ndarray, translation: np.ndarray) -> Transform:
        return Transform.from_rot_trans(Quaternion.from_matrix(rot_matrix), translation)

    def load_transform_from_json(self, path: pathlib.Path, apply_delta: bool = False, key: str | None = None, allow_scale: bool = False) -> Transform | tuple[Transform, np.ndarray]:
        """Load Transform from JSON. If apply_delta=True, adjust translation by delta_x, delta_y."""
        with path.open("r") as f:
            data = json.load(f)
        if key is not None:
            data = data[key]
        rot_matrix = np.array(data["rotation"])
        translation = np.array(data["translation"]).squeeze(axis=-1)

        if apply_delta:
            delta_x = data.get("relative_error", {}).get("delta_x", 0) or 0
            delta_y = data.get("relative_error", {}).get("delta_y", 0) or 0
            translation += np.array([delta_x, delta_y, 0])

        if allow_scale:
            R, S = polar_decompose_RS(np.array(data["rotation"]))
            return self._to_transform(R, translation), np.diag(S)
        else:
            if abs((matrix_det := np.linalg.det(rot_matrix)) - 1) > 1e-5:
                raise RuntimeError(f"Rotation matrix in {path} has det {matrix_det}, not 1")
            return self._to_transform(rot_matrix, translation)

    def __getitem__(self, key: tuple[str, str]) -> Transform | None:
        parent, child = key
        return self.coord_system[parent, child]

    def __call__(self, parent: str, child: str) -> Callable[[np.ndarray], np.ndarray]:
        def apply(x: np.ndarray) -> np.ndarray:
            match (parent, child):
                case ('inf_cam', _):
                    return self.inf_cam2lidar(self['inf_lidar', child](x))
                case (_, 'inf_cam'):
                    return self[parent, 'inf_lidar'](self.inf_lidar2cam(x))
                case _, _:
                    return self[parent, child](x)

        return apply
