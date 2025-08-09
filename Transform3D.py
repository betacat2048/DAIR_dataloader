"""
The transform library for 3-D points transform and coordinate transformation.
CUHK AIoT Lab Kaicheng XIAO
version: 1.1.0 2025/08/07
"""

import torch
import einops
import numpy as np
from dataclasses import dataclass
from typing import Union, Optional, Literal


class NumpyTorchBackend:
    def __init__(self, torch_backend: bool):
        self.torch = torch_backend

    def broadcast_to_match(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.torch:
            target_shape = torch.broadcast_shapes(a.shape, b.shape)

            def _expand(x):
                return x.view((1,) * (len(target_shape) - x.ndim) + x.shape).expand(target_shape)
        else:
            target_shape = np.broadcast_shapes(a.shape, b.shape)

            def _expand(x):
                return np.broadcast_to(x.reshape((1,) * (len(target_shape) - x.ndim) + x.shape), target_shape)

        return _expand(a), _expand(b)

    def can_broadcast(self, a, b) -> bool:
        try:
            if self.torch:
                _ = torch.broadcast_shapes(a.shape, b.shape)
            else:
                _ = np.broadcast_shapes(a.shape, b.shape)
            return True
        except (RuntimeError, ValueError):
            return False

    def stack(self, arrays, dim: int = -1):
        return torch.stack(arrays, dim=dim) if self.torch else np.stack(arrays, axis=dim)

    def concat(self, arrays, dim: int = -1):
        return torch.cat(arrays, dim=dim) if self.torch else np.concatenate(arrays, axis=dim)

    def cross(self, a, b, dim: int = -1):
        return torch.cross(*self.broadcast_to_match(a, b), dim=dim) if self.torch else np.cross(a, b, axis=dim)

    def sum(self, a, dim: int = -1, keepdim: bool = True):
        return torch.sum(a, dim=dim, keepdim=keepdim) if self.torch else np.sum(a, axis=dim, keepdims=keepdim)

    def norm(self, q_arr, dim: int = -1, keepdim: bool = True):
        """Compute length of quaternion(s)"""
        norm_sq = self.sum(q_arr * q_arr, dim=dim, keepdim=keepdim)
        return torch.sqrt(norm_sq) if self.torch else np.sqrt(norm_sq)

    def normalized(self, q_arr):
        """Return unit quaternion(s)"""
        return q_arr / self.norm(q_arr)

    def normalize_(self, q_arr):
        """Return unit quaternion(s)"""
        q_arr /= self.norm(q_arr)
        return q_arr

    def diagonal(self, a, dim1=-2, dim2=-1):
        return torch.diagonal(a, dim1=dim1, dim2=dim2) if self.torch else np.diagonal(a, axis1=dim1, axis2=dim2)

    def argmax(self, a, dim=-1):
        return torch.argmax(a, dim=dim) if self.torch else np.argmax(a, axis=dim)

    def zeros_like(self, a):
        return torch.zeros_like(a) if self.torch else np.zeros_like(a)

    def ones_like(self, a):
        return torch.ones_like(a) if self.torch else np.ones_like(a)

    def sqrt(self, x):
        return torch.sqrt(x) if self.torch else np.sqrt(x)

    def cos(self, x):
        return torch.cos(x) if self.torch else np.cos(x)

    def sin(self, x):
        return torch.sin(x) if self.torch else np.sin(x)

    def deg2rad(self, x):
        return torch.deg2rad(x) if self.torch else np.deg2rad(x)

    def einsum(self, equation: str, *operands):
        return torch.einsum(equation, *operands) if self.torch else np.einsum(equation, *operands)

    def __eq__(self, other):
        return self.torch == other.torch

    def all_close(self, a, b, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan) if self.torch else np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def check_array(self, arr):
        if self.torch != isinstance(arr, torch.Tensor):
            raise TypeError(f'Expected {self}, but got {type(arr)}')

    def to_tensor(self, arr, device=None):
        if self.torch:
            return arr.to(device).float()
        return torch.as_tensor(arr, device=device).float()

    def to_numpy(self, arr):
        if self.torch:
            return arr.numpy(force=True)
        return arr

    def __str__(self) -> str:
        return 'torch' if self.torch else 'numpy'


class Quaternion:
    """
    Quaternion representation: [..., 4] (w, x, y, z).
    Instantiate with either a NumPy array or a PyTorch tensor.

    !!! NOTICE !!!
    1. Quaternion only reference input would NOT clone for performance. (change on input array/tensor may change the Quaternion)
    2. Quaternion would NOT normalize input, please ensure the input is already normalized!
    3. * means compose all times, @ (and call) means apply if the RHS is a vector
        Q @ v ==> v
        Q * Q = Q @ Q ==> Q
        v * Q = Quaternion(v) @ Q = Q * v = Q @ Quaternion(v) ==> Q  (v * Q cannot work for numpy)
    4. Compose None would return a copy of self
    ===
    :param
    q: Quaternion [..., 4] (w, x, y, z). **Assume it already normalized**
    """

    def __init__(self, q: np.ndarray | torch.Tensor):
        # Determine backend
        self.backend = NumpyTorchBackend(not isinstance(q, np.ndarray))

        # Validate shape
        if q.shape[-1] != 4:
            raise ValueError(f"Quaternion must have shape (..., 4), but got {q.shape}")

        self.q = q

    @property
    def s(self):
        return self.q[..., :1]

    @property
    def v(self):
        return self.q[..., 1:]

    @property
    def matrix(self) -> np.ndarray | torch.Tensor:
        """Convert quaternion to rotation matrix property"""
        w, x, y, z = self.q[..., 0:1], self.q[..., 1:2], self.q[..., 2:3], self.q[..., 3:4]
        # common terms
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return self.backend.stack(
            [
                self.backend.concat([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
                self.backend.concat([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
                self.backend.concat([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
            ], dim=-2
        )

    @staticmethod
    def from_matrix(R: np.ndarray | torch.Tensor, eps: float = 1e-8) -> 'Quaternion':
        backend = NumpyTorchBackend(torch_backend=isinstance(R, torch.Tensor))
        if R.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrix must have shape (..., 3, 3)")

        # get the w^2, x^2, y^2, z^2 from the diag of matrix
        diag = backend.diagonal(R)
        m00, m11, m22 = diag[..., 0], diag[..., 1], diag[..., 2]
        w2 = 1.0 + m00 + m11 + m22
        x2 = 1.0 + m00 - m11 - m22
        y2 = 1.0 - m00 + m11 - m22
        z2 = 1.0 - m00 - m11 + m22

        # found the max value for numerical stable
        max_idx = backend.argmax(backend.stack([w2, x2, y2, z2], dim=-1), dim=-1)
        mask_w, mask_x, mask_y, mask_z = (max_idx == 0), (max_idx == 1), (max_idx == 2), (max_idx == 3)

        # result
        q = backend.zeros_like(einops.rearrange(R, '... i j -> ... (i j)')[..., :4])

        yz_a_wx, yz_s_wx = R[..., 2, 1], R[..., 1, 2]
        xz_a_wy, xz_s_wy = R[..., 0, 2], R[..., 2, 0]
        xy_a_wz, xy_s_wz = R[..., 1, 0], R[..., 0, 1]

        # w largest
        if mask_w.any():
            double_w = backend.sqrt(w2[mask_w] + eps)
            q[mask_w, 0] = double_w / 2
            q[mask_w, 1] = (yz_a_wx[mask_w] - yz_s_wx[mask_w]) / (2 * double_w)
            q[mask_w, 2] = (xz_a_wy[mask_w] - xz_s_wy[mask_w]) / (2 * double_w)
            q[mask_w, 3] = (xy_a_wz[mask_w] - xy_s_wz[mask_w]) / (2 * double_w)

        # x largest
        if mask_x.any():
            double_x = backend.sqrt(x2[mask_x] + eps)
            q[mask_x, 0] = (yz_a_wx[mask_x] - yz_s_wx[mask_x]) / (2 * double_x)
            q[mask_x, 1] = double_x / 2
            q[mask_x, 2] = (xy_a_wz[mask_x] + xy_s_wz[mask_x]) / (2 * double_x)
            q[mask_x, 3] = (xz_a_wy[mask_x] + xz_s_wy[mask_x]) / (2 * double_x)

        # y largest
        if mask_y.any():
            double_y = backend.sqrt(y2[mask_y] + eps)
            q[mask_y, 0] = (xz_a_wy[mask_y] - xz_s_wy[mask_y]) / (2 * double_y)
            q[mask_y, 1] = (xy_a_wz[mask_y] + xy_s_wz[mask_y]) / (2 * double_y)
            q[mask_y, 2] = double_y / 2
            q[mask_y, 3] = (yz_a_wx[mask_y] + yz_s_wx[mask_y]) / (2 * double_y)

        # z largest
        if mask_z.any():
            double_z = backend.sqrt(z2[mask_z] + eps)
            q[mask_z, 0] = (xy_a_wz[mask_z] - xy_s_wz[mask_z]) / (2 * double_z)
            q[mask_z, 1] = (xz_a_wy[mask_z] + xz_s_wy[mask_z]) / (2 * double_z)
            q[mask_z, 2] = (yz_a_wx[mask_z] + yz_s_wx[mask_z]) / (2 * double_z)
            q[mask_z, 3] = double_z / 2

        return Quaternion(backend.normalized(q))

    @staticmethod
    def from_euler(yaw: np.ndarray | torch.Tensor, pitch: np.ndarray | torch.Tensor, roll: np.ndarray | torch.Tensor, order: str = 'xyz') -> 'Quaternion':
        backend = NumpyTorchBackend(torch_backend=isinstance(yaw, torch.Tensor))
        y, p, r = backend.deg2rad(yaw), backend.deg2rad(pitch), backend.deg2rad(roll)
        ang = {
            'x': Quaternion(backend.stack([backend.cos(r / 2), backend.sin(r / 2), backend.zeros_like(r), backend.zeros_like(r)])),
            'y': Quaternion(backend.stack([backend.cos(p / 2), backend.zeros_like(r), backend.sin(p / 2), backend.zeros_like(r)])),
            'z': Quaternion(backend.stack([backend.cos(y / 2), backend.zeros_like(r), backend.zeros_like(r), backend.sin(y / 2)])),
        }
        return ang[order[0]] * ang[order[1]] * ang[order[2]]

    def normalized(self) -> 'Quaternion':
        # normalize the length
        new_q = self.backend.normalized(self.q)
        # ensure the direction is positive
        mask = new_q[..., 0] < 0
        new_q[mask] = -new_q[mask]
        return Quaternion(new_q)

    def inverse(self) -> 'Quaternion':
        """Quaternion inverse"""
        return Quaternion(self.backend.concat([self.s, -self.v], dim=-1))

    def compose(self, other: 'Quaternion') -> 'Quaternion':
        if self.backend != other.backend:
            raise TypeError(f"Backends of both quaternions must match, but got {self.backend} and {other.backend}")

        return Quaternion(
            self.backend.concat(
                [
                    self.s * other.s - self.backend.sum(self.v * other.v, dim=-1, keepdim=True),
                    self.v * other.s + other.v * self.s + self.backend.cross(self.v, other.v, dim=-1)
                ], dim=-1
            )
        )

    def apply(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Rotate vector(s) v (..., 3) by this quaternion using cross-product formula"""
        self.backend.check_array(x)
        if x.shape[-1] != 3:
            raise ValueError(f"vector must have shape (..., 3), but got {x.shape}")

        t = 2 * self.backend.cross(self.v, x)
        return x + self.s * t + self.backend.cross(self.v, t)

    def __mul__(self, other: Optional[Union['Quaternion', np.ndarray, torch.Tensor]]) -> 'Quaternion':
        match other:
            case None:
                return Quaternion(self.q)
            case _ if isinstance(other, Quaternion):
                return self.compose(other)
            case _ if isinstance(other, (np.ndarray, torch.Tensor)):
                ''' Transform vector(s) to quaternion(s), then multiply by this quaternion '''
                return self.compose(Quaternion(other))
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __rmul__(self, other: Optional[Union['Quaternion', np.ndarray, torch.Tensor]]) -> 'Quaternion':
        match other:
            case None:
                return Quaternion(self.q)
            case _ if isinstance(other, (np.ndarray, torch.Tensor)):
                return Quaternion(other).compose(self)
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __matmul__(self, other: Optional[Union['Quaternion', np.ndarray, torch.Tensor]]) -> Union['Quaternion', np.ndarray, torch.Tensor]:
        match other:
            case None:
                return Quaternion(self.q)
            case _ if isinstance(other, Quaternion):
                return self.compose(other)
            case _ if isinstance(other, (np.ndarray, torch.Tensor)):
                ''' Apply this rotation to vector(s) '''
                return self.apply(other)
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __rmatmul__(self, other: Optional[Union['Quaternion', np.ndarray, torch.Tensor]]) -> 'Quaternion':
        match other:
            case None:
                return Quaternion(self.q)
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __call__(self, other: Optional[Union['Quaternion', np.ndarray, torch.Tensor]]) -> Union['Quaternion', np.ndarray, torch.Tensor]:
        return self @ other

    def axis_angle(self) -> tuple[np.ndarray, np.ndarray]:
        qa = self.backend.to_numpy(self.q)  # ensure NumPy array
        angle = np.rad2deg(2 * np.arccos(qa[..., 0]))  # angle in degrees
        axis = qa[..., 1:]  # axis (handle zero angle)
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = np.where(norm == 0, np.array([1.0, 0.0, 0.0]), axis / np.where(norm == 0, 1, norm))
        return axis, angle

    def __repr__(self) -> str:
        """Represent quaternion in axis-angle (degrees) form using NumPy for readability"""
        axis, angle = self.axis_angle()
        return f"Quaternion(axis={axis.tolist()}, angle={angle.tolist()}°)"

    def all_close(self, other: 'Quaternion', rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        return self.backend.all_close(self.normalized().q, other.normalized().q, rtol, atol)


class Transform:
    """
    Transform representation: [..., 7] (q0, q1, q2, q3, x, y, z).
    Instantiate with either a NumPy array or a PyTorch tensor.

    !!! NOTICE !!!
    1. Transform only reference input, would NOT clone for performance. (change on input array/tensor may change the Transform)
    2. Transform would NOT normalize the quaternion part, please ensure the quaternion part is already normalized!
    3. * means compose all times, @ (and call) means apply if the RHS is a vector
        T @ v ==> v
        T * T = T @ T ==> T
        v * T = Transform(v) @ T = T * v = T @ Transform(v) ==> T  (v * T cannot work for numpy)
    4. Compose None would return a copy of self
    ===
    :param
    p: Transform [..., 7] (q0, q1, q2, q3, x, y, z). **Assume quaternion part already normalized**
    """

    def __init__(self, p: np.ndarray | torch.Tensor):
        # Determine backend
        self.backend = NumpyTorchBackend(not isinstance(p, np.ndarray))

        # Validate shape
        if p.shape[-1] != 7:
            raise ValueError(f"Transform must have shape (..., 7), but got {p.shape}")

        self.p = p

    @property
    def q(self) -> np.ndarray | torch.Tensor:
        return self.p[..., 0:4]

    # noinspection PyPep8Naming
    @property
    def R(self) -> Quaternion:
        return Quaternion(self.q)

    @property
    def t(self) -> np.ndarray | torch.Tensor:
        return self.p[..., 4:7]

    @property
    def matrix(self) -> np.ndarray | torch.Tensor:
        return self.backend.concat(
            [
                self.backend.concat([self.R.matrix, self.t[..., :, None]], dim=-1),
                self.backend.concat([self.backend.zeros_like(self.t[..., None, :]), self.backend.ones_like(self.t[..., None, :1])], dim=-1),
            ], dim=-2
        )

    @staticmethod
    def from_matrix(m: np.ndarray | torch.Tensor) -> 'Transform':
        return Transform.from_rot_trans(Quaternion.from_matrix(m[..., :3, :3]), m[..., :3, 3])

    @staticmethod
    def from_rot_trans(quaternion: Quaternion | np.ndarray | torch.Tensor, translation: np.ndarray | torch.Tensor, backend: NumpyTorchBackend | None = None) -> 'Transform':
        if isinstance(quaternion, Quaternion):
            if backend is None:
                backend = quaternion.backend
            quaternion = quaternion.q
        if backend is None:
            backend = NumpyTorchBackend(not isinstance(quaternion, np.ndarray))

        backend.check_array(quaternion)
        if quaternion.shape[-1] != 4:
            raise ValueError(f"quaternion must have shape (..., 4), but got {quaternion.shape}")
        backend.check_array(translation)
        if translation.shape[-1] != 3:
            raise ValueError(f"translation must have shape (..., 3), but got {translation.shape}")

        return Transform(backend.concat([quaternion, translation], dim=-1))

    @staticmethod
    def from_rot(quaternion: Quaternion) -> 'Transform':
        return Transform.from_rot_trans(quaternion, quaternion.backend.zeros_like(quaternion.q[..., :3]), quaternion.backend)

    def normalized(self) -> 'Transform':
        return Transform.from_rot_trans(self.R.normalized(), self.t, backend=self.backend)

    def inverse(self) -> 'Transform':
        inverse_R = self.R.inverse()
        return Transform.from_rot_trans(inverse_R, -inverse_R(self.t), backend=self.backend)

    def compose(self, other: 'Transform') -> 'Transform':
        if self.backend != other.backend:
            raise TypeError(f"Backends of both quaternions must match, but got {self.backend} and {other.backend}")
        return Transform.from_rot_trans(self.R * other.R, self.R(other.t) + self.t, backend=self.backend)

    def apply(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        self.backend.check_array(x)
        if x.shape[-1] != 3:
            raise ValueError(f"vector must have shape (..., 3), but got {x.shape}")
        return self.R(x) + self.t

    def __mul__(self, other: Optional[Union['Transform', Quaternion, np.ndarray, torch.Tensor]]) -> 'Transform':
        match other:
            case None:
                return Transform(self.p)
            case _ if isinstance(other, Transform):
                return self.compose(other)
            case _ if isinstance(other, Quaternion):
                return self.compose(Transform.from_rot(other))
            case _ if isinstance(other, (np.ndarray, torch.Tensor)):
                ''' Transform vector(s) to transform(s), then multiply by this transform '''
                return self.compose(Transform(other))
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __rmul__(self, other: Optional[Union['Transform', Quaternion, np.ndarray, torch.Tensor]]) -> 'Transform':
        match other:
            case None:
                return Transform(self.p)
            case _ if isinstance(other, Quaternion):
                return Transform.from_rot(other).compose(self)
            case _ if isinstance(other, (np.ndarray, torch.Tensor)):
                return Transform(other).compose(self)
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __matmul__(self, other: Optional[Union['Transform', Quaternion, np.ndarray, torch.Tensor]]) -> Union['Transform', np.ndarray, torch.Tensor]:
        match other:
            case None:
                return Transform(self.p)
            case _ if isinstance(other, Transform):
                return self.compose(other)
            case _ if isinstance(other, Quaternion):
                return self.compose(Transform.from_rot(other))
            case _ if isinstance(other, (np.ndarray, torch.Tensor)):
                ''' Apply this transform to vector(s) '''
                return self.apply(other)
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __rmatmul__(self, other: Optional[Union['Transform', Quaternion, np.ndarray, torch.Tensor]]) -> 'Transform':
        match other:
            case None:
                return Transform(self.p)
            case _ if isinstance(other, Quaternion):
                return Transform.from_rot(other).compose(self)
            case _:
                raise TypeError(f"Unsupported type {type(other)}.")

    def __call__(self, other: Optional[Union['Transform', Quaternion, np.ndarray, torch.Tensor]]) -> Union['Transform', np.ndarray, torch.Tensor]:
        return self @ other

    def __repr__(self) -> str:
        axis, angle = self.R.axis_angle()
        t = self.backend.to_numpy(self.t)
        return f"Transform(axis={axis.tolist()}, angle={angle.tolist()}°, t={t.tolist()})"

    def all_close(self, other: 'Transform', rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        return self.R.all_close(other.R, rtol=rtol, atol=atol) and self.backend.all_close(self.t, other.t, rtol=rtol, atol=atol)

    @staticmethod
    def mirror_y(transform: 'Transform', axis: Literal['x', 'y', 'z'] = 'y') -> 'Transform':
        p = transform.p
        s, w, u, v, x, y, z = p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5], p[..., 6]
        s = - s
        match axis:
            case 'x':
                w, x = -w, -x
            case 'y':
                u, y = -u, -y
            case 'z':
                v, z = -v, -z
            case _:
                raise ValueError(f"Unsupported axis {axis}.")
        return Transform(transform.backend.stack([s, w, u, v, x, y, z], dim=-1))


class CoordinateSystem:
    @dataclass(slots=True)
    class FrameNode:
        system: 'CoordinateSystem'
        parent: Optional['CoordinateSystem.FrameNode']
        parent2child: Transform | None = None
        level: int = 0

    def __init__(self, root: str = 'root'):
        """
        Initialize a coordinate system with a single root frame.

        Args:
            root (str): Name of the root frame. Defaults to 'root'.

        Note:
            - The root frame has no parent and its transform is implicitly treated as identity.
            - All later frames will be added relative to this root or its descendants.
        """

        self.frame_nodes: dict[str, CoordinateSystem.FrameNode] = {root: CoordinateSystem.FrameNode(self, None)}

    def add(self, parent: str, name: str, parent2child: Transform):
        """
        Add a new frame to the coordinate system tree.

        Args:
            parent (str): The name of the existing parent frame.
            name (str): The name of the new child frame to be added.
            parent2child (Transform): The transform from the parent frame to the child frame.

        Raises:
            KeyError: If the child name already exists or the parent frame is not found.

        Note:
            - The new frame is inserted as a child node in the kinematic tree.
            - The transform is assumed to be from parent to child (i.e., parent @ parent2child = child).
        """

        if name in self.frame_nodes:
            raise KeyError(f"Frame {name} is already exist in the Coordinate system")

        if parent not in self.frame_nodes:
            raise KeyError(f"Cannot find parent node {parent} in the Coordinate system")

        parent = self.frame_nodes[parent]
        self.frame_nodes[name] = CoordinateSystem.FrameNode(self, parent, parent2child, parent.level + 1)

    def __setitem__(self, name: str, new_parent2child: Transform):
        """
        Update the transform from the parent to a given frame.

        Args:
            name (str): The name of the frame whose parent transform is to be updated.
            new_parent2child (Transform): The new transform from the parent to this frame.

        Raises:
            KeyError: If the specified frame does not exist in the coordinate system.

        Warning:
            - This directly modifies the tree structure. It assumes the parent frame remains unchanged.
            - Use it with caution if other transforms are already derived from this frame.
        """

        if name not in self.frame_nodes:
            raise KeyError(f"Cannot find frame {name} in the Coordinate system")
        self.frame_nodes[name].parent2child = new_parent2child

    def __call__(self, parent: str, child: str) -> Transform | None:
        """
        Query the transform from one frame (parent) to another frame (child).

        Args:
            parent (str): Name of the destination (target) frame.
            child (str): Name of the source frame.

        Returns:
            Transform | None: The transform from the parent frame to the child frame (i.e., T such that parent @ T = child).
                              Returns None only if both frames are identical and no transform is required.

        Raises:
            KeyError: If either the parent or child frame is not found.
        """

        if parent not in self.frame_nodes:
            raise KeyError(f"Cannot find frame {parent} in the Coordinate system")
        if child not in self.frame_nodes:
            raise KeyError(f"Cannot find frame {child} in the Coordinate system")
        u_node, v_node = self.frame_nodes[child], self.frame_nodes[parent]

        u2c: Transform | None = None
        v2p: Transform | None = None

        def join_transform(a: Transform | None, b: Transform | None) -> Transform | None:
            if a is None and b is None:
                return None
            return a @ b

        while u_node.level > v_node.level:
            u2c, u_node = join_transform(u_node.parent2child, u2c), u_node.parent
        while v_node.level > u_node.level:
            v2p, v_node = join_transform(v_node.parent2child, v2p), v_node.parent

        while u_node != v_node:
            u2c, u_node = join_transform(u_node.parent2child, u2c), u_node.parent
            v2p, v_node = join_transform(v_node.parent2child, v2p), v_node.parent

        p2v = None if v2p is None else v2p.inverse()
        if p2v is None and u2c is None:
            return None
        return p2v @ u2c  # u == v

    def __getitem__(self, key: tuple[str, str]) -> Transform | None:
        parent, child = key
        return self(parent, child)
