from pathlib import Path
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from utils.crop import CropBorders

def get_classes():
    """
    Returns the list of class names from the dataset.
    """

    return [path.name for path in Path("data/images/train").iterdir() if path.is_dir()]


def make_tfms(train: bool, model_type: str, img_size: int | None = None):
    base_crop = CropBorders(top_ratio=1/3, side_ratio=0.25)
    
    if model_type in {"mobilenetv3_small", "resnet18", "resnet50", "efficientnet_b0"}:
        img_size = img_size or 224
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        ops = [
            base_crop,
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),          # short side → 224
            transforms.CenterCrop(img_size),      # always 224×224
        ]
    elif model_type in {"mobilefacenet", "arcface_r50"}:
        img_size = img_size or 112
        mean = (0.5, 0.5, 0.5)     # → maps [0,1] to [-1,1]
        std  = (0.5, 0.5, 0.5)
        ops = [
            base_crop,
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]

    if train:
        ops += [
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.RandomHorizontalFlip(),
        ]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
