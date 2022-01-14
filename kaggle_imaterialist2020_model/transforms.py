from __future__ import annotations

import math
import numpy as np
from evaluation.submission import get_new_image_size
from pycocotools import mask as mask_api
from utils import box_utils, mask_utils

from kaggle_imaterialist2020_model.types import COCORLE, COCOAnnotation, Prediction


def _encode_mask_fn(x) -> COCORLE:
    encoded_mask = mask_api.encode(np.asfortranarray(x))
    # In the case of byte type, it cannot be converted to json
    encoded_mask["counts"] = str(encoded_mask["counts"])
    return encoded_mask


def convert_predictions_to_coco_annotations(
    prediction: Prediction,
    image_id: int,
    filename: str,
    score_threshold=0.05,
) -> list[COCOAnnotation]:
    """This is made, modifying a function of the same name in
    /tf_tpu_models/official/detection/evaluation/coco_utils.py

    Parameters
    ----------
    prediction : Prediction
        [description]
    image_id: int
    filename: str
    score_threshold : float, optional
        [description], by default 0.05

    Returns
    -------
    list[COCOAnnotation]
        [description]
    """
    prediction["pred_detection_boxes"] = box_utils.yxyx_to_xywh(
        prediction["pred_detection_boxes"]
    )

    mask_boxes = prediction["pred_detection_boxes"]

    orig_shape = prediction["pred_image_info"][0]
    resize_shape = prediction["pred_image_info"][1]

    if orig_shape[0] > orig_shape[1]:
        o2r = orig_shape[0] / resize_shape[0]
    else:
        o2r = orig_shape[1] / resize_shape[1]

    bbox_indices = np.argwhere(
        prediction["pred_detection_scores"] >= score_threshold
    ).flatten()

    predicted_masks = prediction["pred_detection_masks"][bbox_indices]
    image_masks = mask_utils.paste_instance_masks(
        predicted_masks,
        mask_boxes[bbox_indices].astype(np.float32) * o2r,
        int(orig_shape[0]),
        int(orig_shape[1]),
    )
    binary_masks = (image_masks > 0.0).astype(np.uint8)
    encoded_masks = [_encode_mask_fn(binary_mask) for binary_mask in list(binary_masks)]

    mask_masks = (predicted_masks > 0.5).astype(np.float32)
    mask_areas = mask_masks.sum(axis=-1).sum(axis=-1)
    mask_area_fractions = (mask_areas / np.prod(predicted_masks.shape[1:])).tolist()
    mask_mean_scores = (
        (predicted_masks * mask_masks).sum(axis=-1).sum(axis=-1) / mask_areas
    ).tolist()

    anns: list[COCOAnnotation] = []
    for m, k in enumerate(bbox_indices):
        mask_mean_score = mask_mean_scores[m]
        # mask_mean_score is float("nan") when mask_area is 0.
        if not math.isnan(mask_mean_score):
            ann = COCOAnnotation(
                image_id=image_id,
                filename=filename,
                category_id=int(prediction["pred_detection_classes"][k]),
                # Avoid `astype(np.float32)` because
                # it can't be serialized as JSON.
                bbox=tuple(
                    float(x) for x in prediction["pred_detection_boxes"][k] / eval_scale
                ),
                mask_area_fraction=float(mask_area_fractions[m]),
                score=float(prediction["pred_detection_scores"][k]),
                segmentation=encoded_masks[m],
                mask_mean_score=mask_mean_score,
            )
            anns.append(ann)

    return anns
