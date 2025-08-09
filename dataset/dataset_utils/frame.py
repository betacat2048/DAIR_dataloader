import os
import cv2
import torch
import pathlib
import numpy as np
import os.path as osp
from abc import ABC, abstractmethod

from dataset.dataset_utils import read_jpg, load_json
from dataset.v2x_utils.transformation_utils import CoordTransformation_xkc


class Frame(dict, ABC):
    def __init__(self, path, info_dict):
        super().__init__()

        self.path = pathlib.Path(path)
        for key in info_dict:
            self.__setitem__(key, info_dict[key])

    @property
    def calib_camera_intrinsic_path(self) -> pathlib.Path:
        return self.path / self['calib_camera_intrinsic_path']

    @property
    def image_path(self) -> pathlib.Path:
        return self.path / self['image_path']

    def check_data_files(self) -> bool:
        return self.calib_camera_intrinsic_path.exists() and self.image_path.exists()

    def load_calib(self):
        data = load_json(self.calib_camera_intrinsic_path)
        return np.array(data['cam_K'], dtype=np.float64).reshape(3, 3), np.array(data['cam_D'], dtype=np.float64)

    def image(self):
        K, D = self.load_calib()
        img = read_jpg(self.image_path.as_posix())
        h, w = img.shape[:2]

        return img, K, (h, w)

        # new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        # map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=K, distCoeffs=D, R=np.eye(3), newCameraMatrix=new_K, size=(w, h), m1type=cv2.CV_16SC2)
        # img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        # return img, new_K, (h, w)


class VehFrame(Frame):
    def __init__(self, path, veh_dict, tmp_key="tmps"):
        super().__init__(path, veh_dict)
        self.id = {
            "lidar": veh_dict["pointcloud_path"][-10:-4],
            "camera": veh_dict["image_path"][-10:-4]
        }


class InfFrame(Frame):
    def __init__(self, path, inf_dict, tmp_key="tmps"):
        super().__init__(path, inf_dict)
        self.id = {
            "lidar": inf_dict["pointcloud_path"][-10:-4],
            "camera": inf_dict["image_path"][-10:-4]
        }


class VICFrame(Frame):
    def __init__(self, path, info_dict, veh_frame, inf_frame, time_diff):
        super().__init__(path, info_dict)
        self.veh_frame = veh_frame
        self.inf_frame = inf_frame
        self.time_diff = time_diff

        self.coords = CoordTransformation_xkc(
            self.path,
            self.inf_frame,
            self.veh_frame,
        )

    def check_data_files(self) -> bool:
        return self.veh_frame.check_data_files() and self.inf_frame.check_data_files()
