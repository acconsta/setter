import os
import pandas as pd
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils, config_util
from object_detection.builders import model_builder

LABEL_FILE = "data/labels_setter_2022-02-23-01-41-47.json"
LABEL_FILE2 = "data/labels_setter_2022-03-01-05-53-13.json"
SUBLABEL_FILE = "data/labels_setter_2022-02-24-12-00-24.csv"
SUBLABEL_FILE2 = "data/labels_setter_2022-03-01-06-29-47.csv"
TEST_FILES = ["PXL_20220223_015905189.MP.jpg"]
IMAGE_DIR = "data/"

LABELS = dict(
    color=["red", "green", "purple"],
    shape=["diamond", "oval", "squiggle"],
    number=["one", "two", "three"],
    texture=["solid", "striped", "open"],
)


def build_and_restore_model(model_name, num_classes, image_size):
    tf.keras.backend.clear_session()
    print("Building model and restoring weights for fine-tuning...", flush=True)
    pipeline_config = f"/home/tensorflow/models/research/object_detection/configs/tf2/{model_name}.config"
    checkpoint_path = (
        "/home/tensorflow/models/research/object_detection/test_data/checkpoint/ckpt-0"
    )

    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs["model"]
    # print(model_config.centure_net.feature_extractor)
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    nms = (
        model_config.ssd.post_processing.batch_non_max_suppression.use_class_agnostic_nms
    ) = True
    nms.iou_threshold = 0.4
    nms.max_detections_per_class = 26
    nms.max_total_detections = 26
    detection_model = model_builder.build(model_config=model_config, is_training=True)

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor,
    )
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, *image_size, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print("Weights restored!")
    return detection_model


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


def _load_dataframe(label_file, sublabel_file):
    with open(label_file) as f:
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
    box_df = pd.read_csv(sublabel_file, names=["annotation_id", "annotation"])
    box_df["annotation_id"] = box_df.annotation_id.apply(
        lambda s: s.split("_")[-1].split(".")[0]
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
    box_df = box_df.drop("annotation", axis=1).set_index("annotation_id")

    # Add numerical labels
    def get_factor(col):
        return col.apply(lambda s: LABELS[col.name].index(s))

    factors = box_df[label_types].copy().apply(get_factor)
    factors = list(np.stack(factors.to_numpy()))
    box_df["labels"] = factors

    df = df.join(box_df, on="annotation_id", how="inner")
    return df


def get_numpy_training_data(df):
    train_images_np = []
    gt_boxes = []
    gt_labels = []
    for file_name, rows in df.groupby(df.file_name):
        if file_name in TEST_FILES:
            # TODO remove hard docded training example
            continue
        image_path = os.path.join(IMAGE_DIR, rows.file_name.iloc[0])
        try:
            train_images_np.append(load_image_into_numpy_array(image_path))
            gt_boxes.append(np.stack(rows.bbox.to_numpy()))
            gt_labels.append(np.stack(rows.labels.to_numpy()))
        except Exception:
            pass
    assert len(train_images_np) == len(gt_boxes) == len(gt_labels)
    return train_images_np, gt_boxes, gt_labels


def load_data_dataframe():
    return pd.concat(
        [
            _load_dataframe(LABEL_FILE, SUBLABEL_FILE),
            _load_dataframe(LABEL_FILE2, SUBLABEL_FILE2),
        ]
    )
