import argparse, sys
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_classes     # your helper that loads label names

MODEL_TYPE = "mobilenetv3_small"                         # "mobilefacenet" / "arcface_r50"
plot_result = True

# --------------------------------------------------
# 1)  Load class names and TorchScript model
# --------------------------------------------------
classes = get_classes()
num_classes = len(classes)
if not classes:
    print("No classes found in 'data/images/train'. Please ensure the dataset is set up correctly.")
    sys.exit(1)

device = torch.device("cpu")
ckpt = torch.load("saves/mobilenetv3_small_people.pt", map_location="cpu")
model = models.mobilenet_v3_small(weights=None)      # fresh backbone
in_feats = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_feats, num_classes)
model.load_state_dict(ckpt["state_dict"])
model.eval()

if MODEL_TYPE == "mobilenetv3_small":
    IMG = 224
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
else:  # mobilefacenet / arcface_r50 style
    IMG = 112
    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)   # maps [0,1] → [-1,1]
    STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

# --------------------------------------------------
# 4)  Run inference
# --------------------------------------------------
for img_path in Path("data/images/train/henri").glob("*.jpg"):
    print(f"Processing {img_path.name}...")
    if not img_path.is_file():
        continue


    # Load and preprocess the image
    pil = Image.open(img_path)
    x   = preprocess(pil)

    with torch.no_grad():
        logits = model(x)
        prob   = torch.softmax(logits, dim=1)[0]
        confidence, idx = torch.max(prob, dim=0)

    result = classes[idx] if idx < len(classes) else f"class_{idx}"
    print(f"Prediction: {result}  (p = {confidence.item():.3f})")
    for conf, idx in zip(prob, range(len(prob))):
        if conf.item() < 0.01:
            continue
        res = classes[idx] if idx < len(classes) else f"class_{idx}"
        print(f"  {res:20s} {conf.item():.3f}")
        

    # --------------------------------------------------
    # 5)  Optional: show the image with overlay
    # --------------------------------------------------
    if plot_result:
        # Convert PIL RGB → OpenCV BGR for display
        img_bgr = cv2.cvtColor(np.array(pil.resize((640, int(640 * pil.height / pil.width)))), 
                            cv2.COLOR_RGB2BGR)
        cv2.putText(img_bgr,
                    f"{result} {confidence.item():.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Inference", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
