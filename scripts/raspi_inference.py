import time, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from picamera2 import Picamera2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_classes     # your helper that loads label names

# --------------------------------------------------
# 1)  Load class names and TorchScript model
# --------------------------------------------------
classes = get_classes()
if not classes:
    print("No classes found in 'data/images/train'. Please ensure the dataset is set up correctly.")
    sys.exit(1)

model = torch.jit.load("saves/mobilenetv3_small_people.ts", map_location="cpu")
model.eval()

# --------------------------------------------------
# 2)  Picamera2 setup
# --------------------------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# --------------------------------------------------
# 3)  Pre-/post-processing helpers
# --------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def preprocess(rgb):
    """RGB uint8 → torch tensor 1×3×224×224, float32, normalised"""
    img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return t

# --------------------------------------------------
# 4)  Main loop
# --------------------------------------------------
while True:
    # Picamera2 delivers an RGB array (H×W×3, uint8)
    frame_rgb = picam2.capture_array()

    x = preprocess(frame_rgb)
    with torch.no_grad():
        logits = model(x)
        prob   = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(prob, dim=0)

    # Decide label
    if conf.item() < 0.60:
        label = "Unknown"
    else:
        label = f"{classes[idx]}  {conf.item():.2f}"

    # Overlay text
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(frame_bgr, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Picamera2 – MobileNet ID", frame_bgr)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

cv2.destroyAllWindows()
picam2.stop()
