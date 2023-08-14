import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from pathlib import Path

if __name__ == "__main__":
    dataset_path = Path("./datasets/dataset_validation")
    mask_path = dataset_path.joinpath("masks")
    anot_visual_path = dataset_path.joinpath("annotations_visual")
    # dataset_path.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(parents=True, exist_ok=True)
    anot_visual_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_path.joinpath("annotations/instances_default.json").__str__(), "r", encoding="utf-8") as f:
        coco = json.load(f)
    width = coco["images"][0]["width"]
    height = coco["images"][0]["height"]
    for image in coco["images"]:
        mask = np.ones((height, width), dtype=int)
        for annotation in [anot for anot in coco["annotations"] if anot["image_id"] == image["id"]]:
            for pts in annotation["segmentation"]:
                cv2.fillPoly(img=mask, pts=[np.reshape(pts, (-1, 2)).astype(int)], color=annotation["category_id"]+1)
                # plt.imshow(mask)
                # plt.show()
        cv2.imwrite(mask_path.joinpath(f"{image['file_name']}").__str__(), mask)
        plt.imsave(anot_visual_path.joinpath(f"{image['file_name']}").__str__(), mask)
        # tmp = load_img(f"{dataset_path}/masks/{image['file_name']}")
    for c in coco["categories"]:
        print(f"label id {c['id']} = {c['name']}")

    print("hotovo")