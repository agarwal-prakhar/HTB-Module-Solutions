
# Lab Write-Up: Skills Assessment 2 - DeepFool on CIFAR-10

## Overview
This challenge requires implementing the **DeepFool** algorithm to generate an untargeted adversarial example on a CIFAR-10 image. The goal is to perturb a baseline image of a "horse" (class 7) such that the classifier predicts *any* other class, while strictly maintaining an $\ell_2$ distance of $\le 3.5$. 

A unique constraint of this challenge is that the $\ell_2$ distance is measured in **normalized space** ($x_{norm} = (x - \mu)/\sigma$), and all pixel values must remain in the $[0,1]$ range.

## Reconnaissance & API Exploration

The challenge provides a white-box API environment:
1. **`GET /challenge`**: Returns the baseline horse image, `original_class=7`, `l2_threshold=3.5`, and the model's exact normalization statistics.
2. **`GET /model/weights`**: Provides the PyTorch `state_dict` for the `CIFAR10CNN` model.
3. **`POST /predict`**: Allows local validation of intermediate adversarial images.
4. **`POST /submit`**: The validation endpoint. It enforces three conditions: valid $32 \times 32 \times 3$ RGB in $[0,1]$, $\ell_2$ distance in normalized space $\le 3.5$, and misclassification (predicted class $\neq 7$).

## Attack Methodology: DeepFool

DeepFool is an iterative algorithm designed to find the **minimal** perturbation to push an image across a decision boundary. It approximates the distance to the nearest decision boundary by linearizing the classifier's output around the current input.

For an untargeted attack, at each iteration, we compute the minimal vector $r_k$ to cross the boundary of the original class $k_0$ and every other class $k$:
$$ r_k = \frac{|f_k(x) - f_{k_0}(x)|}{\| \nabla f_k(x) - \nabla f_{k_0}(x) \|^2_2} (\nabla f_k(x) - \nabla f_{k_0}(x)) $$
We apply the smallest of these perturbations, add a slight overshoot ($\epsilon = 0.02$) to ensure crossing, and iterate until the predicted class changes.

### 1. Operating in Normalized Space
Because the $\ell_2$ constraint is strictly measured in normalized space, the most elegant approach is to run the DeepFool algorithm directly on the normalized tensor $x_{norm}$. By computing gradients of the model's output with respect to $x_{norm}$, the resulting perturbation vector $r$ inherently minimizes the $\ell_2$ distance in that exact space.

### 2. PyTorch Autograd Pitfalls
During implementation, two common PyTorch autograd issues were encountered:
*   **Non-leaf variables**: `x_adv` must be detached and re-attached to the graph at the start of every iteration using `x_norm.clone().detach().requires_grad_(True)`.
*   **Conversion to NumPy**: The final `x_adv` tensor retained `requires_grad=True`, crashing the PNG encoder. This was fixed by calling `x_norm.detach()` before converting back to $[0,1]$ space.

### 3. The Boundary Fragility Pitfall
Standard DeepFool stops the exact moment the image crosses the decision boundary. However, to send the image to the server, it must be saved as an 8-bit PNG. This quantization (`round()`) introduces tiny pixel shifts (e.g., $0.499 \rightarrow 0$). Because the DeepFool perturbation sits exactly on the boundary, this rounding pushes the image *back* into the original class.

**The Fix:** Since the $\ell_2$ budget is massive ($3.5$) and DeepFool only used $\sim 0.81$, we can scale the final perturbation vector by a factor of $3\times$ (up to a safe limit of $80\%$ of the threshold). This pushes the image *deep* into the target class's decision region, making it completely robust to 8-bit quantization.

## Exploitation Steps

1. **Fetch Challenge & Weights**: Download the baseline image and the `cifar10_model_best.pth` weights.
2. **Normalize Input**: Convert the $[0,1]$ image to normalized space: $x_{norm} = (x - \mu)/\sigma$.
3. **Iterate DeepFool**: For 50 iterations, compute gradients for all 10 classes. Find the minimal perturbation to cross any boundary. Apply it with an overshoot of 1.02.
4. **Box Projection**: After each step, unnormalize the image, clip to $[0,1]$, and renormalize.
5. **Scale for Robustness**: Compute the final perturbation $\delta = x_{adv} - x$. Scale $\delta$ to push the $\ell_2$ distance to $\sim 2.4$ (well within the 3.5 budget).
6. **Local Quantization Check**: Quantize the image to 8-bit integers, convert back to float, and run it through the local model to guarantee it is still misclassified.
7. **Submit**: Encode the array to a base64 PNG and submit to the server.

## Proof of Concept (PoC) Script

```python
import os, io, base64, requests, torch, torch.nn as nn, torchvision.transforms as transforms
from PIL import Image
import numpy as np

BASE_URL = "http://154.57.164.82:30967"

def tensor_from_b64_png(b64):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return transforms.ToTensor()(img)

def b64_png_from_tensor(tensor):
    tensor = torch.clamp(tensor, 0, 1).detach()
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def l2_normalized_space(a, b, mean_t, std_t):
    a, b = a.detach(), b.detach()
    return float(torch.norm(((a - mean_t) / std_t) - ((b - mean_t) / std_t)))

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.fc1, self.fc2 = nn.Linear(64*8*8, 128), nn.Linear(128, num_classes)
        self.pool1, self.pool2 = nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)
        self.relu1, self.relu2, self.relu3 = nn.ReLU(), nn.ReLU(), nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc2(self.dropout(self.relu3(self.fc1(x))))

def deepfool(x_orig, mean_t, std_t, model, num_classes=10, overshoot=0.02, max_iter=50):
    x_norm = ((x_orig - mean_t) / std_t).clone()
    with torch.no_grad():
        clean_pred = torch.argmax(model(x_norm.unsqueeze(0))).item()
        
    for i in range(max_iter):
        x_var = x_norm.clone().detach().requires_grad_(True)
        output = model(x_var.unsqueeze(0))[0]
        k0 = torch.argmax(output).item()
        
        if k0 != clean_pred: break
        
        grads = [torch.autograd.grad(output[k], x_var, retain_graph=True)[0] for k in range(num_classes)]
        f_k0, grad_k0 = output[k0], grads[k0]
        
        min_dist, best_r = float('inf'), None
        for k in range(num_classes):
            if k == k0: continue
            w, f_diff = grads[k] - grad_k0, output[k] - f_k0
            w_norm = torch.norm(w)
            if w_norm > 1e-12:
                dist = torch.abs(f_diff) / w_norm
                if dist < min_dist:
                    min_dist = dist
                    best_r = (torch.abs(f_diff) / (w_norm**2)) * w
                    
        if best_r is None: break
        x_norm = x_norm + (1 + overshoot) * best_r
        x_norm = ((torch.clamp(x_norm * std_t + mean_t, 0, 1) - mean_t) / std_t)

    x_adv_final = x_norm.detach() * std_t + mean_t
    return torch.clamp(x_adv_final, 0, 1)

if __name__ == "__main__":
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = tensor_from_b64_png(ch["image"])
    l2_threshold = float(ch["l2_threshold"])
    mean_t = torch.tensor(ch["normalization"]["mean"]).view(3, 1, 1)
    std_t = torch.tensor(ch["normalization"]["std"]).view(3, 1, 1)

    weights_path = "cifar10_model_best.pth"
    if not os.path.exists(weights_path):
        with open(weights_path, "wb") as f: f.write(requests.get(f"{BASE_URL}/model/weights").content)
    
    model = CIFAR10CNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    x_adv = deepfool(x, mean_t, std_t, model)
    
    # Scale perturbation to be robust against 8-bit quantization
    delta = x_adv - x
    l2_norm = l2_normalized_space(x, x_adv, mean_t, std_t)
    if l2_norm > 0:
        scale = min(l2_norm * 3.0, l2_threshold * 0.8) / l2_norm
        x_adv = torch.clamp(x + scale * delta, 0, 1)
        
    r = requests.post(f"{BASE_URL}/submit", json={"image": b64_png_from_tensor(x_adv)}, timeout=10)
    if r.status_code == 200 and r.json().get("success"):
        print(f"FLAG: {r.json().get('flag')}")
```

## Conclusion
By implementing DeepFool directly in the model's normalized space, we efficiently solved the $\ell_2$ constraint. Overcoming the fragility of the minimal perturbation boundary via vector scaling ensured the 8-bit PNG quantization did not revert the image back to its original class.

### Flag
`HTB{d33pf00l_k33p5_0n_f00l1ng}`

--------------------------------------

└──╼ [★]$ python3 solve4012.py
[*] Target: http://154.57.164.82:30967
[*] Original: 7, L2 Threshold (norm space): 3.5
[*] Computing DeepFool perturbation...
[+] DeepFool misclassified at iteration 2
[*] Scaled L2 distance (normalized): 2.443183
[*] Local quantized prediction: 9 (Original: 7)
[*] Submitting to server...

========================================
FLAG: HTB{d33pf00l_k33p5_0n_f00l1ng}
========================================

----------------------------------------

import os
import io
import base64
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

BASE_URL = "http://154.57.164.82:30967"

# ---------- Helpers ----------
def tensor_from_b64_png(b64: str) -> torch.Tensor:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (32, 32):
        raise ValueError("Expected 32x32 PNG")
    return transforms.ToTensor()(img)

def b64_png_from_tensor(tensor: torch.Tensor) -> str:
    tensor = torch.clamp(tensor, 0, 1).detach()
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def l2_normalized_space(a: torch.Tensor, b: torch.Tensor, mean_t, std_t) -> float:
    a = a.detach()
    b = b.detach()
    a_norm = (a - mean_t) / std_t
    b_norm = (b - mean_t) / std_t
    return float(torch.norm(a_norm - b_norm))

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

# ---------- DeepFool Attack ----------
def deepfool(x_orig, mean_t, std_t, model, num_classes=10, overshoot=0.02, max_iter=50):
    x_norm_orig = ((x_orig - mean_t) / std_t).clone()
    x_norm = x_norm_orig.clone()
    
    with torch.no_grad():
        clean_pred = torch.argmax(model(x_norm_orig.unsqueeze(0))).item()
        
    for i in range(max_iter):
        x_norm_var = x_norm.clone().detach().requires_grad_(True)
        output = model(x_norm_var.unsqueeze(0))[0]
        k0 = torch.argmax(output).item()
        
        if k0 != clean_pred:
            print(f"[+] DeepFool misclassified at iteration {i}")
            break
            
        grads = []
        for k in range(num_classes):
            model.zero_grad()
            grad = torch.autograd.grad(output[k], x_norm_var, retain_graph=True)[0]
            grads.append(grad)
            
        f_k0 = output[k0]
        grad_k0 = grads[k0]
        
        min_dist = float('inf')
        best_r = None
        
        for k in range(num_classes):
            if k == k0: continue
            f_k = output[k]
            grad_k = grads[k]
            w = grad_k - grad_k0
            f_diff = f_k - f_k0
            w_norm = torch.norm(w)
            if w_norm > 1e-12:
                dist = torch.abs(f_diff) / w_norm
                if dist < min_dist:
                    min_dist = dist
                    best_r = (torch.abs(f_diff) / (w_norm**2)) * w
                    
        if best_r is None: break
            
        x_norm = x_norm + (1 + overshoot) * best_r
        x_adv = x_norm * std_t + mean_t
        x_adv = torch.clamp(x_adv, 0, 1)
        x_norm = (x_adv - mean_t) / std_t

    x_norm = x_norm.detach()
    x_adv_final = x_norm * std_t + mean_t
    return torch.clamp(x_adv_final, 0, 1)

# ---------- Main Execution ----------
if __name__ == "__main__":
    print(f"[*] Target: {BASE_URL}")
    
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = tensor_from_b64_png(ch["image"])
    orig_class = int(ch["original_class"])
    l2_threshold = float(ch["l2_threshold"])
    
    mean = ch["normalization"]["mean"]
    std = ch["normalization"]["std"]
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    
    print(f"[*] Original: {orig_class}, L2 Threshold (norm space): {l2_threshold}")

    weights_path = "cifar10_model_best.pth"
    if not os.path.exists(weights_path):
        print("[*] Downloading model weights...")
        with open(weights_path, "wb") as f:
            f.write(requests.get(f"{BASE_URL}/model/weights").content)
    
    model = load_model(weights_path, device="cpu")

    print("[*] Computing DeepFool perturbation...")
    x_adv = deepfool(x, mean_t, std_t, model)
    
    # SCALE THE PERTURBATION TO BE ROBUST AGAINST 8-BIT QUANTIZATION
    delta = x_adv - x
    l2_norm = l2_normalized_space(x, x_adv, mean_t, std_t)
    
    if l2_norm > 0:
        # Push the perturbation 3x deeper into the boundary, up to 80% of the max budget
        target_l2 = min(l2_norm * 3.0, l2_threshold * 0.8)
        scale = target_l2 / l2_norm
        x_adv = torch.clamp(x + scale * delta, 0, 1)
        
    print(f"[*] Scaled L2 distance (normalized): {l2_normalized_space(x, x_adv, mean_t, std_t):.6f}")
    
    # VERIFY LOCALLY ON THE QUANTIZED IMAGE
    img_array = (x_adv.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    x_adv_q = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    
    x_adv_q_norm = (x_adv_q - mean_t) / std_t
    with torch.no_grad():
        pred_adv_q = int(torch.argmax(model(x_adv_q_norm.unsqueeze(0))).item())
    print(f"[*] Local quantized prediction: {pred_adv_q} (Original: {orig_class})")

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
