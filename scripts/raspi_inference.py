import torch, cv2
import numpy as np

classes = ["Alice","Bob"]  # or load from your saved classes file
model = torch.jit.load("saves/mobilenetv3_small_people.ts", map_location="cpu")
model.eval()

def preprocess(bgr):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)/255.0
    img = (img - np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0)
    return t

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    x = preprocess(frame)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(prob, dim=0)
    name = classes[idx]
    if conf.item() < 0.6:  # threshold for "unknown"
        label = "Unknown"
    else:
        label = f"{name} {conf.item():.2f}"
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Pi", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cap.release(); cv2.destroyAllWindows()
