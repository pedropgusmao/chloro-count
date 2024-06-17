import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import labelme
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from PIL import Image
from readlif.reader import LifFile
from torch.nn import Module
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

CHLORO_CHLORO_MAX_INTERSECTION = 0.1
BUNDLE_CHLORO_MIN_INTERSECTION = 0.9
MIN_SCORE_CHLORO = 0.0
MIN_SEG_BUNDLE = 0.2


def save_json_file(
    img_path,
    json_path,
    list_predictions: List[np.ndarray],
    label: str,
    append: bool = False,
):
    all_points = []
    write_flag = "w" if not append else "a"
    for pred in list_predictions:
        pred_squeeze = np.squeeze(pred)
        pred_bin = np.zeros(pred_squeeze.shape)
        pred_bin[np.where(pred_squeeze > 0.1)] = 255
        contours, _ = cv2.findContours(
            pred_bin.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1
        )
        for c in contours:  # There should be just one per prediction already
            if c.shape[0] > 5:
                point_list = c[:, 0, :].astype(np.float32)
                if point_list.shape[0] > 20:
                    point_list = point_list[0::2]
                if point_list.shape[0] > 20:
                    point_list = point_list[0::2]
                inner_dict = {
                    "label": label,
                    "points": point_list.tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
                all_points.append(inner_dict)

    img = labelme.LabelFile.load_image_file(img_path)
    if img is not None:
        img_data = base64.b64encode(img).decode("utf-8")
        out_dict = {
            "version": "5.0.1",
            "flags": {},
            "shapes": all_points,
            "imagePath": str(img_path),
            "imageData": img_data,
            "imageHeight": 512,
            "imageWidth": 512,
        }
        with open(json_path, write_flag) as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=2)


def iou(img_a: npt.NDArray, img_b: npt.NDArray) -> float:
    binary_a = img_a > 0
    binary_b = img_b > 0
    intersection = np.count_nonzero(np.logical_and(binary_a, binary_b))
    union = np.count_nonzero(np.logical_or(binary_a, binary_b))
    return float(intersection) / union


def non_max_suppression(
    segmented_candidates: npt.NDArray,
    max_rel_inter: float = CHLORO_CHLORO_MAX_INTERSECTION,
):
    # https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
    # good idea to sort segmented_candidates by score pos0 being highest
    num_candidates = len(segmented_candidates)
    if num_candidates == 1:
        return segmented_candidates
    P = [segmented_candidates[j] for j in range(num_candidates)]
    keep = []
    if len(P) == 0:
        return []
    else:
        S = P.pop()
        keep.append(S)
        while len(P) > 1:
            P = [T for T in P if iou(S, T) < max_rel_inter]
            if len(P) > 0:
                S = P.pop()
                keep.append(S)
    return np.stack(keep)


def get_instance_segmentation_model(num_classes: int) -> Module:
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def extract_images_and_scales_from_lif(
    lif_image_path: Path, extraction_root: Path = Path("./extracted_images")
):
    lif_file = LifFile(lif_image_path)
    lif_dir = extraction_root / lif_image_path.stem
    cell_img_list = [i for i in lif_file.get_iter_image()]
    for cell in cell_img_list:
        cell_dir = lif_dir / cell.name
        cell_dir.mkdir(parents=True, exist_ok=True)

        # Save the scales
        with open(cell_dir / "scale.csv", "w") as f:
            f.write(f"{abs(cell.scale[0])}, {abs(cell.scale[1])}, {abs(cell.scale[2])}")

        # Save slices
        images_dir = cell_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for slice_idx in range(cell.nz):
            slice_img_channels = [
                img_channel for img_channel in cell.get_iter_c(t=0, z=int(slice_idx))
            ]
            slice_pil_img = Image.merge("RGB", slice_img_channels)
            slice_pil_img.save(images_dir / f"{slice_idx}.png")


def clean_min_score(
    prediction_masks: npt.NDArray,
    prediction_scores: npt.NDArray,
    score_threshold: float = MIN_SCORE_CHLORO,
) -> npt.ArrayLike:
    good_segmentations = prediction_scores >= score_threshold
    return prediction_masks[good_segmentations]


def filter_across_planes(
    slices_in_cell: List[Tuple[npt.ArrayLike, npt.ArrayLike]],
    score_threshold: float,
    non_max_supp_threshold: float,
):
    # Filter predictions based on prediction scores
    predictions_filt_score = [
        clean_min_score(pred, score, score_threshold=score_threshold)
        for (pred, score) in slices_in_cell
    ]

    # Filter predictions based on non_max_suppression.
    # This will make sure no two suggested chloroplasts intersect
    predictions_filt_non_max = [
        non_max_suppression(pred, max_rel_inter=non_max_supp_threshold)
        for pred in predictions_filt_score
    ]
    return predictions_filt_non_max


def segment_single_cell(
    cell_dir: Path,
    model: Module,
    device: torch.device,
    thresholds: Dict[str, float],
    label: str = "chloroplast",
):
    images_dir = cell_dir / "images"
    slices_in_cell = []
    for image_path in sorted(images_dir.iterdir()):
        image = Image.open(image_path)
        image_batch = torch.stack([torch.tensor(np.array(image))])
        image_batch = image_batch.permute(0, 3, 1, 2).float() / 255.0
        image_batch = image_batch.to(device)
        output_predictions = model(image_batch)
        slices_in_cell.append(
            (
                output_predictions[0]["masks"].cpu().numpy(),
                output_predictions[0]["scores"].cpu().numpy(),
            )
        )

    # Filter planes based on prediction scores and non-max suppression
    filtered_segments = filter_across_planes(
        slices_in_cell,
        score_threshold=thresholds["score_threshold"],
        non_max_supp_threshold=thresholds["non_max_suppression"],
    )
    json_dir = cell_dir / "annotations"
    json_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(sorted(images_dir.iterdir())):
        json_path = json_dir / f"{image_path.stem}.json"
        save_json_file(image_path, json_path, filtered_segments[idx], label=label)

    return filtered_segments


def segment_cells(
    extraction_dir: Path,
    model_path: Path,
    threshold_dict: Dict[str, float],
    label: str = "chloroplast",
):
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # Load the images
    with torch.no_grad():
        for lif_dir in extraction_dir.iterdir():
            for cell_dir in lif_dir.iterdir():
                output_predictions = segment_single_cell(
                    cell_dir, model, device, threshold_dict, label=label
                )
            # output_predictions.save(cell_dir / "masks" / f"{image_path.stem}.png")

    return output_predictions


if __name__ == "__main__":
    threshold_dict = {
        "score_threshold": MIN_SCORE_CHLORO,
        "non_max_suppression": CHLORO_CHLORO_MAX_INTERSECTION,
    }
    # Extract all images from  lifs_dir
    lifs_dir = Path("./data/")
    extraction_dir = Path("./extracted_images")
    for lif_path in lifs_dir.iterdir():
        if lif_path.suffix == ".lif":
            extract_images_and_scales_from_lif(lif_path, extraction_root=extraction_dir)

    # Segment all cells in the extracted images
    segment_cells(
        extraction_dir=extraction_dir,
        model_path=Path("./models/chloroplasts.pth"),
        threshold_dict=threshold_dict,
        label="chloroplast",
    )
