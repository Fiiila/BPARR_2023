import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import load_img

if __name__ == "__main__":
    dataset_path = f"./datasets/dataset_01_segmentation"
    with open(f"{dataset_path}/annotations/instances_default.json", "r", encoding="utf-8") as f:
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
        cv2.imwrite(f"{dataset_path}/masks/{image['file_name']}", mask)
        plt.imsave(f"{dataset_path}/annotations_visual/{image['file_name']}", mask)
        tmp = load_img(f"{dataset_path}/masks/{image['file_name']}")

    print("hotovo")