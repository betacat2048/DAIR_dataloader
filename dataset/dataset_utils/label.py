import pathlib
import numpy as np

from dataset.v2x_utils import get_3d_8points
from dataset.dataset_utils import load_json
from Transform3D import Quaternion, Transform, CoordinateSystem


class Label(dict):
    def __init__(self, path: pathlib.Path, world2label: Transform | None = None, rot_recompense: Quaternion | None = None, label_filter=None):
        super().__init__()

        boxes_type = []
        boxes_dim = []
        boxes_pose = []
        boxes_corners = []

        for label in load_json(path):
            obj_type = label['type'].lower()

            dim = label["3d_dimensions"]
            dim = np.array([float(dim["w"]), float(dim["l"]), float(dim["h"])])  # (x, y, z) <==> (w, l, h)

            rot = Quaternion.from_euler(*(np.rad2deg(np.array([float(label.get("rotation", np.nan)), 0, 0])) + np.array([90, 0, 0])))  # the rot of target (add 90 in yaw as different definition of vehicle coord)
            if rot_recompense is not None:
                rot = rot_recompense * rot

            pos = label["3d_location"]
            pos = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"]) - float(dim[-1]) / 2])

            pose = Transform.from_rot_trans(rot, pos)
            if world2label is not None:
                pose = world2label(pose)

            corners = np.array(label["world_8_points"]) if "world_8_points" in label else get_3d_8points(dim, pose)

            # determine if box is in extended range
            if label_filter is None or label_filter(corners):
                boxes_type.append(obj_type)
                boxes_dim.append(dim)
                boxes_pose.append(pose.p)
                boxes_corners.append(corners)

        self['obj_type'] = boxes_type
        self['boxes_dim'] = np.stack(boxes_dim, axis=0)
        self['boxes_pose'] = np.stack(boxes_pose, axis=0)
        self['boxes_corners'] = np.stack(boxes_corners, axis=0)
