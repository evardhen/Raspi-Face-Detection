from insightface.model_zoo import get_model
from models.mobilefacenet import MobileFaceNet
import torch.nn as nn
import torch
from torchvision import models
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_classes


def load_model(model_type, num_classes):
    if model_type == "mobilefacenet":
        img_size = 112
        scripted = torch.jit.load("models/mobilefacenet_scripted.pt", map_location="cpu")
        sd = scripted.state_dict()
        model = MobileFaceNet(num_classes=num_classes)
        msd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
    elif model_type == "arcface_r50":
        backbone = get_model("arcface_r50", download=True)
        in_feats = 512
        img_size = 112
        head = nn.Linear(in_feats, num_classes, bias=True)
        model = nn.Sequential(backbone, head) 
    elif model_type == "mobilenetv3_small":
        # Load pretrained MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        # Replace classifier for our classes
        in_feats = model.classifier[3].in_features  # last Linear
        img_size = 224
        model.classifier[3] = nn.Linear(in_feats, num_classes)
    return model, img_size

def probe_model(path):
    # Try TorchScript first
    try:
        model = torch.jit.load(path, map_location="cpu")
        print("Loaded as TorchScript.")
        kind = "torchscript"
    except Exception as e:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.nn.Module):
            print("This is a pickled nn.Module (not TorchScript). "
                "You need the original Python class to import.")
            kind = "pickled_module"
            model = obj
        else:
            print("This is a checkpoint/state_dict. You need the model class to rebuild.")
            kind = "state_dict"
            model = None
    return model, kind

if __name__ == "__main__":
    model, kind = probe_model("models/mobilefacenet_scripted.pt")
    if model is not None:
        print(f"Model loaded successfully as {kind}.")
        print(f"Classes: {get_classes()}")
    else:
        print(f"Failed to load model. Kind: {kind}.")