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


class InstanceObj:
    def __init__(
        self,
        *,
        class_label: str,
        instance_id: int,
        num_slices: int,
        current_slice: int = 0,
    ):
        self.class_label: str = class_label
        self.instance_id: int = instance_id
        self.num_slices: int = num_slices
        self.current_slice = current_slice
        self.list_segments: List[npt.NDArray] = [[] for _ in range(num_slices)]

    def append_segment(self, segment: npt.NDArray):
        self.list_segments[self.current_slice] = segment
        self.current_slice += 1

    def get_latest_segment(self) -> npt.NDArray:
        latest_segment = self.list_segments[0]
        for idx in reversed(range(self.num_slices)):
            latest_segment = self.list_segments[idx]
            if len(latest_segment) != 0:
                break

        return latest_segment

    def get_instance_id(self) -> int:
        return self.instance_id

    def close(self):
        num_current_slices = len(self.list_segments)
        for j in range(num_current_slices, self.num_slices):
            self.list_segments.append([])


def save_instance_segmentation_json(
    path_original_annotations, path_instance_annotations, total_dict
):
    num_slices = len(list(path_original_annotations.glob("*.json")))
    for slice_idx in range(0, num_slices):
        # Get original image information:
        json_original = path_original_annotations / f"{slice_idx}.json"
        with open(json_original, "r") as f:
            original_data = json.load(f)

        json_instance = path_instance_annotations / f"{slice_idx}.json"
        all_points = []
        for instance_id, instance in total_dict.items():
            point_list = instance.list_segments[slice_idx]
            if len(point_list):
                inner_dict = {
                    "label": f"{instance_id}",
                    "points": point_list.tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
                all_points.append(inner_dict)
        out_dict = {
            "version": "5.0.1",
            "flags": {},
            "shapes": all_points,
            "imagePath": original_data["imagePath"],
            "imageData": original_data["imageData"],
            "imageHeight": 512,
            "imageWidth": 512,
        }

        with open(json_instance, "w") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=2)


def save_json_file(
    img_path,
    json_path,
    list_predictions: List[np.ndarray],
    label: str,
    append: bool = False,
):
    all_points = []
    # If append and json_path exists, load the json file and append the new predictions
    list_predictions_and_labels = [(list_predictions, label)]
    if append and json_path.exists():
        with open(json_path, "r") as f:
            tmp_json = json.load(f)
            all_points += tmp_json["shapes"]

    for list_predictions, label in list_predictions_and_labels:
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
        with open(json_path, "w") as f:
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
) -> npt.NDArray:
    good_segmentations = prediction_scores >= score_threshold
    return prediction_masks[good_segmentations]


def filter_across_planes(
    slices_in_cell: List[Tuple[npt.NDArray, npt.NDArray]],
    score_threshold: float,
    non_max_supp_threshold: float,
) -> List[npt.NDArray]:
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
    append_annotations: bool = False,
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
        save_json_file(
            image_path,
            json_path,
            filtered_segments[idx],
            label=label,
            append=append_annotations,
        )

    return filtered_segments


def get_segments_from_labelme_json(json_file: Path, class_label: str) -> npt.NDArray:
    """Extracts segments from a labelme json file

    Args:
        json_file (Path): Complete path to json file containing labelme segments.
        class_label (str): Specific label of segments to be extracted.

    Returns:
        ArrayLike:
    """
    list_segments = []
    with open(json_file, "r") as f:
        data = json.load(f)
        for segment in data["shapes"]:
            if segment["label"] == class_label:
                contours = np.array(segment["points"], dtype=np.int32)
                list_segments.append(contours)
    return list_segments


def distance_segment_instances_with_id(segment_centers, instances_centers):
    dist_matrix = np.zeros(shape=(len(segment_centers), len(instances_centers)))
    # Create matrix
    for i, Ps in enumerate(segment_centers):
        Ps = np.array(list(Ps))
        for j, Pc in enumerate(instances_centers):
            Pc = np.array(list(Pc))
            dist_matrix[i, j] = np.linalg.norm(Ps - Pc)

    return dist_matrix


def get_center_from_poly(poly_seg: npt.NDArray) -> Tuple[float, float]:
    M = cv2.moments(poly_seg)
    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]
    return (cX, cY)


def get_centers_from_seg(list_segments: List[npt.NDArray]) -> List[Tuple[float, float]]:
    list_centers = []
    for poly_seg in list_segments:
        cX, cY = get_center_from_poly(poly_seg)
        list_centers.append((cX, cY))
    return list_centers


def get_centers_from_instance_objects(
    instance_objs: List[InstanceObj],
) -> List[Tuple[float, float]]:
    list_centers = []
    for instance in instance_objs:
        latest_poly_seg = instance.get_latest_segment()
        list_centers.append(get_center_from_poly(latest_poly_seg))
    return list_centers


def associate_segments(
    segments: List[npt.NDArray], instances: List[InstanceObj], dist_th: float
):
    associated_instances = []
    while segments and instances:
        segments_centers = get_centers_from_seg(segments)
        instances_centers = get_centers_from_instance_objects(instances)
        D = distance_segment_instances_with_id(segments_centers, instances_centers)
        min_dist = np.amin(D)
        if min_dist < dist_th:
            x, y = np.where(D == min_dist)
            this_segment = segments.pop(int(x[0]))
            this_instance = instances.pop(int(y[0]))
            this_instance.append_segment(this_segment)
            associated_instances.append(this_instance)
        else:
            break
    remaining_segments = segments

    return associated_instances, remaining_segments


def update_instance_dict(
    instances_dict, remaining_segments, class_label, num_slices, current_slice
):
    num_current_instances = len(instances_dict)
    new_instances = []
    for idx, segment in enumerate(remaining_segments):
        instance = InstanceObj(
            class_label=class_label,
            instance_id=idx + num_current_instances,
            num_slices=num_slices,
            current_slice=current_slice,
        )
        instance.append_segment(segment)
        k = f"{class_label}_{instance.instance_id}"
        instances_dict[k] = instance
        new_instances.append(instance)

    return instances_dict, new_instances


def process_instance_seg_for_class(json_dir: Path, class_label: str, max_dist: float):
    instances_dict = {}
    num_slices = len(list(json_dir.glob("*.json")))
    previous_instances = []
    for slice_idx in range(num_slices):
        print(slice_idx)
        json_file = json_dir / f"{slice_idx}.json"
        current_segs = get_segments_from_labelme_json(
            json_file, class_label=class_label
        )
        associated_instances, remaining_segments = associate_segments(
            segments=current_segs, instances=previous_instances, dist_th=max_dist
        )
        instances_dict, new_instances = update_instance_dict(
            instances_dict, remaining_segments, class_label, num_slices, slice_idx
        )
        previous_instances = associated_instances + new_instances
    return instances_dict


def segment_cells(
    extraction_dir: Path,
    model_path: Path,
    threshold_dict: Dict[str, float],
    label: str = "chloroplast",
    append_annotations=True,
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
                    cell_dir,
                    model,
                    device,
                    threshold_dict,
                    label=label,
                    append_annotations=append_annotations,
                )
            # output_predictions.save(cell_dir / "masks" / f"{image_path.stem}.png")

    return output_predictions


def save_instances_volumes(volumes_dict, path_to_cell):
    path_to_file = path_to_cell / "volumes.txt"
    with open(path_to_file, "w") as f:
        for k, v in volumes_dict.items():
            f.write(f"{k},{v}\n")


def calculate_volume_from_instances(instance_dict, scales):
    volume_dict = {}
    dx, dy, dz = scales
    for instance_id, instance in instance_dict.items():
        total_area = 0.0
        for cnt in instance.list_segments:
            if len(cnt) != 0:  # considers continuous
                total_area += cv2.contourArea(cnt) * dx * dy
        volume_dict[instance_id] = total_area * dz

    return volume_dict


def get_scale_from_lif_file(path_to_scales_csv: Path):
    with open(path_to_scales_csv, "r") as f:
        x_px_um, y_px_um, z_px_um = f.read().split(",")
    return (1.0 / float(x_px_um), 1.0 / float(y_px_um), abs(1.0 / float(z_px_um)))


def instance_association(extraction_dir: Path, dist_dict: Dict[str, float]):
    for lif_dir in extraction_dir.iterdir():
        for cell_dir in lif_dir.iterdir():
            scales = get_scale_from_lif_file(cell_dir / "scale.csv")
            if cell_dir.is_dir():
                path_instance_annotations = cell_dir / "instance_annotations"
                path_instance_annotations.mkdir(exist_ok=True, parents=True)
                try:
                    path_adjusted_annotations = cell_dir / "annotations"
                    instances_dict_chloro = process_instance_seg_for_class(
                        path_adjusted_annotations,
                        class_label="chloroplast",
                        max_dist=dist_dict["chloroplast"],
                    )
                    instances_dict_bundle = process_instance_seg_for_class(
                        path_adjusted_annotations,
                        class_label="bundle sheath",
                        max_dist=dist_dict["bundle sheath"],
                    )
                    total_dict = {**instances_dict_chloro, **instances_dict_bundle}
                    save_instance_segmentation_json(
                        path_adjusted_annotations, path_instance_annotations, total_dict
                    )
                    volumes_dict = calculate_volume_from_instances(total_dict, scales)
                    save_instances_volumes(volumes_dict, cell_dir)

                except Exception as e:
                    print(f"Error with {cell_dir}")
                    print(e)


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
            # extract_images_and_scales_from_lif(lif_path, extraction_root=extraction_dir)
            pass

    # Segment all cells in the extracted images
    """segment_cells(
        extraction_dir=extraction_dir,
        model_path=Path("./models/chloroplast.pth"),
        threshold_dict=threshold_dict,
        label="chloroplast",
    )
    segment_cells(
        extraction_dir=extraction_dir,
        model_path=Path("./models/bundle_sheath.pth"),
        threshold_dict=threshold_dict,
        label="bundle sheath",
        append_annotations=True,
    )"""

    # Instance segmentation for chloroplasts
    dist_dict = {"chloroplast": 10.0, "bundle sheath": 100.0}
    instance_association(Path("./extracted_images_clean"), dist_dict)
