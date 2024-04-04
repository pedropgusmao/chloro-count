from typing import List, Tuple
import numpy as np
import torch
import torchvision
from PIL import Image
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.nn import Module
from readlif.reader import LifFile


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
    lif_image_path: Path, extracted_dir: Path = Path("./extracted_images")
):
    lif_file = LifFile(lif_image_path)
    lif_dir = extracted_dir / lif_image_path.stem
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


def filter_across_planes(slices_in_cell: List[Tuple[torch.Tensor, torch.Tensor]]):
    # Filter the masks across planes
    # For each mask, check if the mask is present in the other planes
    # If it is, keep it, otherwise, discard it
    # If the mask is present in all planes, keep it
    # If the mask is present in some planes, keep it if the mask is present in at least 2 planes
    # If the mask is present in only one plane, discard it
    # If the mask is not present in any plane, discard it
    pass


def segment_images(extracted_dir: Path, model_path: Path):
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # Load the images
    with torch.no_grad():
        for lif_dir in extracted_dir.iterdir():
            for cell_dir in lif_dir.iterdir():
                images_dir = cell_dir / "images"
                slices_in_cell = []
                for image_path in images_dir.iterdir():
                    image = Image.open(image_path)
                    image_batch = torch.stack([torch.tensor(np.array(image))])
                    image_batch = image_batch.permute(0, 3, 1, 2).float() / 255.0
                    image_batch = image_batch.to(device)
                    output_predictions = model(image_batch)
                    slices_in_cell.append(
                        (
                            output_predictions[0]["masks"],
                            output_predictions[0]["scores"],
                        )
                    )
            # output_predictions.save(cell_dir / "masks" / f"{image_path.stem}.png")

    return output_predictions


if __name__ == "__main__":
    extract_images_and_scales_from_lif(Path("./2314510#7.lif"))
    segment_images(
        extracted_dir=Path("./extracted_images"),
        model_path=Path("./models/chloroplasts.pth"),
    )
