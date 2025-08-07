import torch
from torchvision import models

ckpt = torch.load("saves/mobilenetv3_small_people.pt", map_location="cpu")
model = models.mobilenet_v3_small(weights=None)
in_feats = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_feats, len(ckpt["classes"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

example = torch.randn(1,3,224,224)
ts = torch.jit.trace(model, example)
ts.save("saves/mobilenetv3_small_people.ts")