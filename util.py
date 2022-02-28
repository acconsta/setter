import os
import pandas as pd
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from object_detection.utils import visualization_utils as viz_utils

LABEL_FILE = "data/labels_setter_2022-02-23-01-41-47.json"
SUBLABEL_FILE = "data/labels_setter_2022-02-24-12-00-24.csv"
TRAIN_IMAGE_DIR = "data/train"

LABELS = dict(
    color=["red", "green", "purple"],
    shape=["diamond", "oval", "squiggle"],
    number=["one", "two", "three"],
    texture=["solid", "striped", "open"],
)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    with open(path, "rb") as f:
        img_data = f.read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def plot_detections(
    image_np,
    boxes,
    classes,
    scores,
    category_index,
    figsize=(12, 16),
    ax=None,
    image_name=None,
):
    """Wrapper function to visualize detections.

    Args:
      image_np: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      figsize: size for the figure.
      image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.5,
    )
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        if ax is not None:
            ax.imshow(image_np_with_annotations)
        else:
            plt.figure(figsize=figsize)
            plt.imshow(image_np_with_annotations)


def convert_to_tf_boxes(bbox, height, width):
    # x, y, width, height
    a, b, c, d = bbox
    c += a
    d += b
    a, b, c, d = a / width, b / height, c / width, d / height
    assert all([a <= 1, b <= 1, c <= 1, d <= 1]), bbox
    return np.array([b, a, d, c], dtype=np.float32)


def load_data_dataframe():
    with open(LABEL_FILE) as f:
        labels = json.load(f)
    images_df = pd.DataFrame(labels["images"]).set_index("id")
    annotations_df = pd.DataFrame(labels["annotations"]).rename(
        columns={"id": "annotation_id"}
    )
    df = images_df.join(annotations_df.set_index("image_id"))

    # Load images and bounding boxes
    def convert(row):
        row["bbox"] = convert_to_tf_boxes(row["bbox"], row["height"], row["width"])
        return row

    df = df.apply(convert, axis=1)

    # Load class labels for each bounding box
    box_df = pd.read_csv(SUBLABEL_FILE, names=["annotation_id", "annotation"])
    box_df["annotation_id"] = box_df.annotation_id.apply(
        lambda s: s.split("_")[1].split(".")[0]
    ).astype("int")
    label_types = list(LABELS.keys())

    def parse_label_list(l: str, label_type):
        l = l.strip("[]").lower().split(",")
        matches = [l for l in l if l in LABELS[label_type]]
        assert len(matches) == 1, (l, label_type)
        return matches[0]

    for label_type in label_types:
        box_df[label_type] = box_df.annotation.apply(
            lambda s: parse_label_list(s, label_type)
        ).astype("category")
    box_df = box_df.drop("annotation", 1).set_index("annotation_id")

    # Add numerical labels
    def get_factor(col):
        return col.apply(lambda s: LABELS[col.name].index(s))

    factors = box_df[label_types].copy().apply(get_factor)
    factors = list(np.stack(factors.to_numpy()))
    box_df["labels"] = factors

    df = df.join(box_df, on="annotation_id")
    return df

def get_numpy_training_data(df):
    train_images_np = []
    gt_boxes = []
    gt_labels = []
    for _, rows in df.groupby(df.index):
        image_path = os.path.join(TRAIN_IMAGE_DIR, rows.file_name.iloc[0])
        try:
            train_images_np.append(load_image_into_numpy_array(image_path))
            gt_boxes.append(np.stack(rows.bbox.to_numpy()))
            gt_labels.append(np.stack(rows.labels.to_numpy()))
        except Exception:
            pass
    assert len(train_images_np) == len(gt_boxes) == len(gt_labels)
    return train_images_np, gt_boxes, gt_labels


get_numpy_training_data(load_data_dataframe())

