from utils import (
    extract_images_and_scales_from_lif,
    segment_cells,
    MIN_SCORE_CHLORO,
    CHLORO_CHLORO_MAX_INTERSECTION,
)
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lif_dir",
        type=str,
        required=True,
        default="./data/",
        help="Path to directory containing all LIF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="./extracted_images/",
        help="Directory where images and segmentations will created.",
    )
    parser.add_argument(
        "--dir_models",
        type=str,
        required=True,
        default="./models/",
        help="Directory containing models used for segmentation. Notice that the name of the model will define the segments' labels.",
    )
    parser.add_argument(
        "--skip_image_extraction",
        type=bool,
        required=False,
        default=False,
        help="Skip image extraction step and just regenerate segmentations.",
    )
    args = parser.parse_args()
    lifs_dir = Path(args.lif_dir)
    extraction_dir = Path(args.output_dir)
    dir_models = Path(args.dir_models)

    threshold_dict = {
        "score_threshold": MIN_SCORE_CHLORO,
        "non_max_suppression": CHLORO_CHLORO_MAX_INTERSECTION,
    }

    # Extract all images from  lifs_dir if needed
    if not args.skip_image_extraction:
        for lif_path in lifs_dir.iterdir():
            if lif_path.suffix == ".lif":
                extract_images_and_scales_from_lif(
                    lif_path, extraction_root=extraction_dir
                )

    # Segment all cells in the extracted images
    for path_to_model in dir_models.iterdir():
        if path_to_model.suffix == ".pth":
            label = path_to_model.stem
            segment_cells(
                extraction_dir=extraction_dir,
                model_path=path_to_model,
                threshold_dict=threshold_dict,
                label=label,
            )
