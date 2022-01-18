from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from kaggle_imaterialist2020_model.cmd.segment import main as segment
from PIL import Image
from segmentation.transforms import coco_rle_to_mask

resource_dir = Path(__file__).parents[2] / "tests/resources"
image_dir = resource_dir / "images"
expected_mask_dir = resource_dir / "masks"


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    rle["counts"] = rle["counts"].lstrip("b").strip("'").replace("\\\\", "\\").encode()
    mask = coco_rle_to_mask(rle)
    # {0, 1}^(heght, width)
    return mask


def iou(a: np.ndarray, e: np.ndarray) -> np.float64:
    return np.logical_and(a, e).sum() / np.logical_or(a, e).sum()


IMAGE = np.array(Image.open(image_dir / "00a8764cff12b2e849c850f4be5608bc.jpg"))


def save_mask_image(mask, out):
    t = 0.7
    plt.imshow(((1 - t) * IMAGE + t * 255 * mask[:, :, None]).astype(np.uint8))
    plt.axis("off")
    plt.savefig(out, bbox_inches="tight")


def join_cateogry(df):
    # TODO: Include the following categories into training artifacts
    # and dynamicaly load them from the config at this script.
    # pasted from https://github.com/hrsma2i/dataset-iMaterialist/blob/main/raw/classes.txt
    df_c = (
        pd.Series(
            [
                "shirt|blouse",
                "top|t-shirt|sweatshirt",
                "sweater",
                "cardigan",
                "jacket",
                "vest",
                "pants",
                "shorts",
                "skirt",
                "coat",
                "dress",
                "jumpsuit",
                "cape",
                "glasses",
                "hat",
                "headband|head_covering|hair_accessory",
                "tie",
                "glove",
                "watch",
                "belt",
                "leg_warmer",
                "tights|stockings",
                "sock",
                "shoe",
                "bag|wallet",
                "scarf",
                "umbrella",
            ]
        )
        .reset_index()
        .rename(columns={"index": "category_id", 0: "category"})
    )

    df = df.merge(df_c, on="category_id")

    return df


def main(
    config_file: str = typer.Option(
        ...,
        help="A config YAML file to load a trained model. "
        "Choose from GCS URI (gs://bucket/models/foo/config.yaml) or local path (path/to/config.yaml).",
    ),
    checkpoint_path: str = typer.Option(
        ...,
        help="A Tensorflow checkpoint file to load a trained model. "
        "Choose from GCS URI (gs://bucket/models/foo/model.ckpt-1234) or local path (path/to/model.ckpt-1234).",
    ),
    out_dir: Path = typer.Option(
        ...,
        help="The directory path to save segmentation.jsonlines"
        " and images for qualitative evaluation.",
    ),
) -> None:
    """Check that editing the training code (tf_tpu_models/official/detection/main.py)
    doesn't make the accuracy worse.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "segmentation.jsonlines"
    segment(
        config_file=config_file,
        checkpoint_path=checkpoint_path,
        image_dir=str(image_dir),
        cache_dir=str(out_dir / "saved_model"),
        out=str(out_json),
        resize=640,
    )

    print("load segmentation")
    df_act = pd.read_json(out_json, lines=True)
    df_act = join_cateogry(df_act)
    actual_masks = df_act["segmentation"].apply(rle_to_mask)

    mask_image_dir = out_dir / "mask_images"
    mask_image_dir.mkdir(parents=True, exist_ok=True)
    actual_mask_dir = out_dir / "actual_masks"
    actual_mask_dir.mkdir(parents=True, exist_ok=True)
    print(f"save actual mask images at: {mask_image_dir}")
    df_act["mask"] = actual_masks

    def _save_mask_and_image(row):
        suffix = f"{row['index']}_{row['category']}"
        save_mask_image(
            row["mask"],
            mask_image_dir / f"actual_{suffix}.png",
        )
        # to update test cases in mask_dir
        np.save(
            actual_mask_dir / f"{suffix}.npy",
            row["mask"],
        )

    df_act.reset_index().apply(_save_mask_and_image, axis=1)

    print("check each expected mask exists in the actual masks")
    for mask_file in expected_mask_dir.glob("*.npy"):

        expected = np.load(mask_file)

        save_mask_image(expected, mask_image_dir / f"expected_{mask_file.stem}.png")

        assert actual_masks.apply(
            lambda actual: iou(actual, expected) > 0.90
        ).any(), f"{mask_file.name} mask doesn't exist in the prediction."

        print(f"{mask_file}: OK")


if __name__ == "__main__":
    typer.run(main)
