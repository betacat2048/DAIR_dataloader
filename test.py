import os.path
import pathlib
import logging
import numpy as np
import torch.utils.data

from dataset.v2x_utils import RectFilter, get_3d_8points, draw_bboxes_points_and_wireframe
from dataset.dataset_utils import load_json, InfFrame, VehFrame, VICFrame, Label

import torch
import matplotlib.pyplot as plt
from Transform3D import Quaternion, Transform, CoordinateSystem


def build_path_to_info(prefix, data, sensortype="lidar"):
    path2info = {}
    if sensortype == "lidar":
        for elem in data:
            if elem["pointcloud_path"] == "":
                continue
            path = (pathlib.Path(prefix) / elem["pointcloud_path"]).as_posix()
            path2info[path] = elem
    elif sensortype == "camera":
        for elem in data:
            if elem["image_path"] == "":
                continue
            path = (pathlib.Path(prefix) / elem["image_path"]).as_posix()
            path2info[path] = elem
    return path2info


def get_split(split_path, split, frame_pairs):
    if os.path.exists(split_path):
        split_data = load_json(split_path)
    else:
        print("Split File Doesn't Exists!")
        raise Exception

    if split in ["train", "val", "test"]:
        split_data = split_data["cooperative_split"][split]
    else:
        print("Split Method Doesn't Exists!")
        raise Exception

    frame_pairs_split = []
    for frame_pair in frame_pairs:
        veh_frame_idx = frame_pair["vehicle_image_path"].split("/")[-1].replace(".jpg", "")
        if veh_frame_idx in split_data:
            frame_pairs_split.append(frame_pair)
    return frame_pairs_split


class VICDataset(torch.utils.data.Dataset):
    def __init__(self, path, split="train", extended_range=None):
        super().__init__()
        self.path = pathlib.Path(path)

        # get path info of each frame
        self.inf_path2info = build_path_to_info(
            "infrastructure-side",
            load_json(self.path / "infrastructure-side/data_info.json"),
            'camera',
        )
        self.veh_path2info = build_path_to_info(
            "vehicle-side",
            load_json(self.path / "vehicle-side/data_info.json"),
            'camera',
        )

        # read in the frame pair and data splits
        frame_pairs = load_json(self.path / "cooperative/data_info.json")
        split_path = "data/split_datas/example-cooperative-split-data.json"
        frame_pairs = get_split(split_path, split, frame_pairs)

        self.data = []
        # process each frame
        for elem in frame_pairs:
            # read in the frame info
            inf_frame = self.inf_path2info[elem["infrastructure_image_path"]]
            veh_frame = self.veh_path2info[elem["vehicle_image_path"]]

            # build frame class
            inf_frame = InfFrame(self.path / "infrastructure-side", inf_frame)
            veh_frame = VehFrame(self.path / "vehicle-side", veh_frame)
            vic_frame = VICFrame(self.path, elem, veh_frame, inf_frame, 0)

            # get the labels
            filter_world = None if extended_range is None else RectFilter(vic_frame.coords['world', 'veh_lidar'](extended_range).squeeze(axis=0))
            label_v = Label(self.path / elem["cooperative_label_path"], rot_recompense=vic_frame.coords['world', 'veh_lidar'].R, label_filter=filter_world)
            # label_v = Label(self.path / 'infrastructure-side/label/virtuallidar/' / f"{inf_frame.id['lidar']}.json", world2label=vic_frame.coords['world', 'inf_lidar'], rot_recompense=None, label_filter=filter_world)

            if vic_frame.check_data_files():
                self.data.append((vic_frame, label_v,))

    def __getitem__(self, index):
        raise NotImplementedError


class VICSyncDataset(VICDataset):
    def __init__(self, path, split="train", extended_range=None):
        super().__init__(path=path, split=split, extended_range=extended_range, )
        logging.info("VIC-Sync {} dataset, overall {} frames".format(split, len(self.data)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dair_dataset = VICSyncDataset(
        path="E:\\DAIR\\DAIR-V2X-C",
        split='train',
    )

    for vic_frame, label in dair_dataset:
        bbox_pts = get_3d_8points(label['boxes_dim'], Transform(label['boxes_pose']))
        # bbox_pts = label['boxes_corners']

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 12))

        ax2.plot(*vic_frame.coords['world', 'veh_cam'](np.array([[0, 0, 0], [10, 0, 10], [-10, 0, 10], [0, 0, 0]]))[..., :2].T, 'r-', label='Vehicle')
        ax2.plot(*vic_frame.coords['world', 'veh_cam'].t[:2], 'ro')
        ax2.plot(*vic_frame.coords('world', 'inf_cam')(np.array([[0, 0, 0], [10, 0, 10], [-10, 0, 10], [0, 0, 0]]))[..., :2].T, 'b-', label='Infrastructure')
        ax2.plot(*vic_frame.coords('world', 'inf_cam')(np.zeros(3))[:2], 'bo')
        for tag in Transform(label['boxes_pose'][..., None, :])(np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [0, 1, 0]]) * label['boxes_dim'][..., None, :])[..., :2]:
            ax2.plot(*tag.T, 'c-')
        ax2.set_aspect('equal')
        ax2.legend()

        # veh size
        img, K, (h, w) = vic_frame.veh_frame.image()
        ax3.imshow(img)

        vic_frame: VICFrame
        p = K @ vic_frame.coords('world', 'inf_cam')(np.zeros(3))
        p = (p[..., :2] / p[..., 2:])[(p[..., 2] > 0)]
        p = p[(-0.2 * w < p[..., 0]) & (p[..., 0] < 1.2 * w) & (-0.2 * h < p[..., 1]) & (p[..., 1] < 1.2 * h)]
        ax3.plot(*p.T, 'rx')

        draw_bboxes_points_and_wireframe(vic_frame.coords['veh_cam', 'world'](bbox_pts), K, ax3, w, h)

        # infra size
        img, K, (h, w) = vic_frame.inf_frame.image()
        ax1.imshow(img)

        p = K @ vic_frame.coords('inf_cam', 'veh_cam')(np.zeros(3))
        p = (p[..., :2] / p[..., 2:])[(p[..., 2] > 0)]
        p = p[(-0.2 * w < p[..., 0]) & (p[..., 0] < 1.2 * w) & (-0.2 * h < p[..., 1]) & (p[..., 1] < 1.2 * h)]
        ax1.plot(*p.T, 'rx')

        draw_bboxes_points_and_wireframe(vic_frame.coords('inf_cam', 'world')(bbox_pts), K, ax1, w, h)

        plt.show()

        pass
