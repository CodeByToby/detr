"""
Builder function for a custom database.

Modified from the coco.py file and with help from https://gist.github.com/woctezuma/e9f8f9fe1737987351582e9441c46b5d
"""

from pathlib import Path
from .coco import CocoDetection, make_coco_transforms

def make_custom_transforms(image_set):
# Taken from the coco.py file

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([300, 450, 600]), # More aggressive zoom
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Adds visual diversity
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "annotations" / f'train.json'),
        "val": (root / "val", root / "annotations" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_custom_transforms(image_set), return_masks=args.masks)
    return dataset