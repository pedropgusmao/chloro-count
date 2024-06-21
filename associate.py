from utils import instance_association
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_and_segments_dir",
        type=str,
        required=True,
        default="./extracted_images/",
        help="Directory where images and segmentations were created.",
    )
    args = parser.parse_args()
    extraction_dir = Path(args.output_dir)

    # Instance segmentation for chloroplasts
    dist_dict = {"chloroplast": 10.0, "bundle sheath": 100.0}
    instance_association(extraction_dir, dist_dict)
