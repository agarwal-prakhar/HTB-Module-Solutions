
# Lab Write-Up: Skills Assessment 1 - Targeted I-FGSM on CIFAR-10

## Overview
This challenge marks a step up in complexity from the previous MNIST tasks. It requires a **targeted adversarial attack** on a color image (CIFAR-10) using a more sophisticated CNN architecture. The objective is to perturb a baseline image of a "dog" (class 5) such that the classifier predicts "cat" (class 3), while strictly maintaining an $L_\infty$ distance of $\epsilon = 8/255 \approx 0.031373$ in the $[0,1]$ pixel space.

## Reconnaissance & API Exploration

The challenge provides a white-box environment via a REST API:
1. **`GET /challenge`**: Returns the baseline dog image, `original_class=5`, `target_class=3`, `epsilon=8/255`, and the model's exact normalization statistics (mean and std).
2. **`GET /model/weights`**: Provides the PyTorch `state_dict` for the `CIFAR10CNN` model, allowing local gradient computation.
3. **`POST /predict`**: Allows local validation of intermediate adversarial images.
4. **`POST /submit`**: The validation endpoint. It enforces three conditions:
   - The image is a valid $32 \times 32 \times 3$ RGB image in $[0,1]$.
   - The $L_\infty$ distance to the baseline is $\le \epsilon$.
   - The model's predicted class equals the target class (3).

## Attack Methodology: Targeted I-FGSM

Standard FGSM takes a single large step in the direction of the gradient's sign. However, for a tight $L_\infty$ budget like $8/255$ and a specific target class, a single step is often insufficient to cross the complex, non-linear decision boundary of a CNN. 

Instead, we use **Iterative FGSM (I-FGSM)**, also known as Basic Projected Gradient Descent (PGD). 

### The Targeted I-FGSM Algorithm
For a targeted attack, we want to *minimize* the loss with respect to the target class $y_{target}$, effectively pulling the image into the target's decision boundary. The update rule at each step $t$ is:
$$ x_{adv}^{t+1} = \text{Clip}_{x, \epsilon} \left( x_{adv}^{t} - \alpha \cdot \text{sign}(\nabla_x J(x_{adv}^{t}, y_{target})) \right) $$

Where:
*   $\alpha$ is the step size (we use $1/255$, ensuring fine-grained progress).
*   $J$ is the CrossEntropy loss.
*   $\text{Clip}_{x, \epsilon}$ projects the image back into the $L_\infty$ ball of radius $\epsilon$ around the original image $x$, and clamps values to $[0,1]$.

### Model Normalization
The `CIFAR10CNN` expects normalized inputs. Because the model's `forward` pass does not apply normalization internally (unlike the MNIST challenges), we must apply it manually before passing the image to the model: 
`x_norm = (x - mean) / std`
We compute the gradient with respect to the unnormalized $[0,1]$ image, as that is the space where the $L_\infty$ constraint is enforced.

### 8-bit Quantization Safety
To prevent PNG 8-bit integer quantization from pushing pixels slightly outside the $L_\infty$ budget (e.g., $0.031373$ rounding up to $9/255 = 0.035294$), we explicitly use `.round()` when converting the tensor back to an integer image for PNG encoding. Because our step size $\alpha = 1/255$ and $\epsilon = 8/255$, the pixel values naturally land exactly on $1/255$ intervals, making the quantization exact.

## Exploitation Steps

1. **Fetch Challenge & Weights**: Download the baseline dog image and the `cifar10_model_best.pth` weights.
2. **Initialize Attack**: Load the model in evaluation mode and initialize `x_adv` as a clone of the original image.
3. **Iterative Loop**: Run for 100 iterations. In each iteration:
   - Apply normalization.
   - Compute the CrossEntropy loss against the target class (3).
   - Compute the gradient of the loss w.r.t the input image.
   - Take a step in the *opposite* direction of the gradient's sign.
   - Project the image back into the $L_\infty$ ball and clip to $[0,1]$.
4. **Submit**: Encode the final tensor to a base64 PNG and submit to the server.

## Proof of Concept (PoC) Script

```python
import os
import io
import base64
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

BASE_URL = "http://154.57.164.82:30842"

# ---------- Helpers ----------
def tensor_from_b64_png(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return transforms.ToTensor()(img)

def b64_png_from_tensor(tensor: torch.Tensor) -> str:
    # .round() ensures strict 8-bit quantization without floating point drift
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------- Model ----------
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------- Main Execution ----------
if __name__ == "__main__":
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = tensor_from_b64_png(ch["image"])
    target_class = int(ch["target_class"])
    epsilon = float(ch["epsilon"])
    
    mean = torch.tensor(ch["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(ch["normalization"]["std"]).view(3, 1, 1)

    weights_path = "cifar10_model_best.pth"
    if not os.path.exists(weights_path):
        with open(weights_path, "wb") as f:
            f.write(requests.get(f"{BASE_URL}/model/weights").content)
    
    model = CIFAR10CNN(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Targeted I-FGSM
    x_adv = x.clone()
    alpha = 1.0 / 255.0
    target_tensor = torch.tensor([target_class])
    
    for i in range(100):
        x_adv.requires_grad = True
        x_norm = (x_adv - mean) / std
        output = model(x_norm.unsqueeze(0))
        
        loss = F.cross_entropy(output, target_tensor)
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # Step opposite to the gradient sign to minimize target loss
        x_adv = x_adv.detach() - alpha * grad.sign()
        
        # Project into L_inf ball and [0,1] bounds
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    r = requests.post(
        f"{BASE_URL}/submit",
        json={"image": b64_png_from_tensor(x_adv)},
        timeout=10
    )
    
    if r.status_code == 200 and r.json().get("success"):
        print(f"FLAG: {r.json().get('flag')}")
```

## Conclusion
By implementing Targeted I-FGSM, we successfully guided the image across the decision boundary from "dog" to "cat" within a strict $8/255$ pixel budget. Managing the internal model normalization and carefully projecting back into the $L_\infty$ ball at each iteration were the key factors in bypassing the server's strict validation checks.

### Flag
`HTB{f1r57_0rd3r_fun}`

-------------------------------

Here is the complete Python script to solve the "Skills Assessment 1" challenge. It implements a **Targeted Iterative FGSM (I-FGSM / PGD)** attack. 

Since we want the model to predict a specific target class (cat), we minimize the CrossEntropy loss with respect to the target label. This means we take steps in the *negative* direction of the gradient's sign. To ensure the final image passes the server's strict $L_\infty$ validation, I've also included strict integer-space quantization and projection.

### `solve.py`

```python
import os
import io
import base64
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

BASE_URL = "http://154.57.164.82:30842"

# ---------- Helpers ----------
def tensor_from_b64_png(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (32, 32):
        raise ValueError("Expected 32x32 PNG")
    return transforms.ToTensor()(img)  # (3, 32, 32) in [0,1]

def b64_png_from_tensor(tensor: torch.Tensor) -> str:
    # Use round() to safely handle 8-bit PNG quantization
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.abs(a - b).max())

# ---------- Model ----------
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(model_path: str, device: str = "cpu") -> CIFAR10CNN:
    model = CIFAR10CNN(num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

# ---------- Main Execution ----------
if __name__ == "__main__":
    print(f"[*] Target: {BASE_URL}")
    
    # 1. Fetch challenge
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = tensor_from_b64_png(ch["image"])
    orig_class = int(ch["original_class"])
    target_class = int(ch["target_class"])
    epsilon = float(ch["epsilon"])
    
    mean = torch.tensor(ch["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(ch["normalization"]["std"]).view(3, 1, 1)
    
    print(f"[*] Original: {orig_class}, Target: {target_class}, Epsilon: {epsilon:.6f}")

    # 2. Download weights
    weights_path = "cifar10_model_best.pth"
    if not os.path.exists(weights_path):
        print("[*] Downloading model weights...")
        with open(weights_path, "wb") as f:
            f.write(requests.get(f"{BASE_URL}/model/weights").content)
    
    model = load_model(weights_path, device="cpu")

    # 3. Targeted I-FGSM Attack
    print("[*] Computing Targeted I-FGSM perturbation...")
    x_adv = x.clone()
    alpha = 1.0 / 255.0  # Step size
    num_iter = 100
    
    target_tensor = torch.tensor([target_class])
    
    for i in range(num_iter):
        x_adv.requires_grad = True
        x_norm = (x_adv - mean) / std
        output = model(x_norm.unsqueeze(0))
        
        # Minimize loss w.r.t to the target class
        loss = F.cross_entropy(output, target_tensor)
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # Step in the negative direction of the gradient sign
        x_adv = x_adv.detach() - alpha * grad.sign()
        
        # Project back into the L_inf ball and [0,1] box
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
    print(f"[*] Final L_inf distance: {linf(x, x_adv):.6f}")
    
    # 4. Check prediction locally
    with torch.no_grad():
        pred = model(((x_adv - mean) / std).unsqueeze(0)).argmax(dim=1).item()
    print(f"[*] Local adversarial prediction: {pred}")

    # 5. Submit to server
    print("[*] Submitting to server...")
    r = requests.post(
        f"{BASE_URL}/submit",
        json={"image": b64_png_from_tensor(x_adv)},
        timeout=10
    )
    
    if r.status_code == 200:
        data = r.json()
        if data.get("success"):
            print("\n" + "="*40)
            print(f"FLAG: {data.get('flag')}")
            print("="*40)
        else:
            print("[-] Validation failed:", data)
    else:
        print(f"[-] Submission failed ({r.status_code}): {r.text}")
```

### Execution Instructions

1. Install the required dependencies:
   ```bash
   pip3 install torch torchvision pillow requests numpy
   ```
2. Save the script above as `solve.py`.
3. Run the script:
   ```bash
   python3 solve.py
   ```

------------------------------

]
└──╼ [★]$ python3 solve30.py
[*] Target: http://154.57.164.82:30842
[*] Original: 5, Target: 3, Epsilon: 0.031373
[*] Downloading model weights...
[*] Computing Targeted I-FGSM perturbation...
[*] Final L_inf distance: 0.031373
[*] Local adversarial prediction: 3
[*] Submitting to server...

========================================
FLAG: HTB{f1r57_0rd3r_fun}
========================================
----------------------------------


