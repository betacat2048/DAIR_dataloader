import json
import yaml
import pickle
import numpy as np
from PIL import Image

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.load(f)
    return data


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)


def read_jpg(jpg_path: str) -> np.ndarray:
    with Image.open(jpg_path) as img:
        img = img.convert('RGB')  # Ensure RGB mode
        image = np.array(img, dtype=np.uint8)  # (H, W, C), same as mmcv.imread
    return image
