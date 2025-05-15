"""
Builder function for a custom database.

Modified from the coco.py file and with help from https://gist.github.com/woctezuma/e9f8f9fe1737987351582e9441c46b5d
"""

from pathlib import Path

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "annotations" / f'train.json'),
        "val": (root / "val", root / "annotations" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset