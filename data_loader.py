import os
import cv2
import numpy as np
from pathlib import Path

def load_images(data_dir_name, labels_dict, img_size=(224, 224)):
    # Assume downloaded dataset is stored in: data/penalty_images/<class_name>/*.JPEG
    data_dir = Path("data") / data_dir_name

    X, y = [], []
    for label_name, label_value in labels_dict.items():
        image_paths = list((data_dir / label_name).glob("*.JPEG"))
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_value)
    return np.array(X), np.array(y)