"""Extract bounding boxes from each image for further annotation."""
import os
from functools import lru_cache
import json
import cv2
import numpy as np

LABEL_FILE = "data/labels_setter_2022-02-23-01-41-47.json"
IMAGE_DIR = "data/"
OUT_DIR = "data/"


@lru_cache(100)
def read_image(type, file_name):
    return cv2.imread(os.path.join(IMAGE_DIR, type, file_name))


with open(LABEL_FILE) as f:
    labels = json.load(f)
print(labels.keys())


def extract_card(image, c):
    rectangle = np.array([[0, 0], [0, 122], [200, 112], [200, 0]])

    # Orient rectangle
    def dist(p1, p2):
        return np.abs(p1 - p2).sum()

    if dist(c[0], c[1]) > dist(c[1], c[2]):
        c = np.concatenate([c[1:], c[:1]])

    try:
        homog = cv2.findHomography(c, rectangle, cv2.RANSAC)
        extract = cv2.warpPerspective(image, homog[0], (200, 122))
        # show(cv2.pyrDown(extract))
    except Exception as e:
        print(e)
        pass
    return extract


for type in ["train", "test"]:
    image_id_file_map = {i["id"]: i["file_name"] for i in labels["images"]}
    for annotation in ["boxes", "polygons"]:
        os.makedirs(os.path.join(OUT_DIR, type, annotation), exist_ok=True)
    for label in labels["annotations"]:
        print(label)
        image = read_image(type, image_id_file_map[label["image_id"]])
        if image is None:
            continue

        id = label["id"]
        box = label["bbox"]
        box = [int(i) for i in box]
        x, y, w, h = box
        crop = image[y : y + h, x : x + w]
        out_name = os.path.join(OUT_DIR, type, "boxes", f"box_{id}.png")
        cv2.imwrite(out_name, crop)

        polgyon = label["segmentation"][0]
        polygon = np.array([int(i) for i in polgyon]).reshape(-1, 2)
        assert polygon.shape[0] == 4

        out_name = os.path.join(OUT_DIR, type, "polygons", f"polygon_{id}.png")
        extract = extract_card(image, polygon)
        cv2.imwrite(out_name, extract)
