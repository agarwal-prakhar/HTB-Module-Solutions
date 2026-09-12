# Lab Write-Up: DeepFool Challenge

## Overview
This challenge advances from untargeted attacks to a **Targeted DeepFool** attack. The goal is to perturb a baseline MNIST image of a digit (originally a `4`) so that the classifier predicts a specific target class (`6`), while strictly maintaining an $\ell_2$ distance of $\le 0.75$ between the original and adversarial images.

## Reconnaissance & API Exploration

The challenge provides a white-box API environment, giving us everything needed to compute gradients locally:
1.  **`GET /challenge`**: Returns the baseline image, `label=4`, `target=6`, and `l2_threshold=0.75`.
2.  **`GET /weights`**: Provides the PyTorch `state_dict` for the `SimpleClassifier` model.
3.  **`POST /predict`**: Allows local validation of intermediate adversarial images.
4.  **`POST /submit`**: The validation endpoint. It verifies the $\ell_2$ distance and whether the model predicts the target class `6`.

The model architecture (`SimpleClassifier`) is a standard PyTorch CNN that takes $28\times28$ images in $[0,1]$ space and outputs log-softmax probabilities.

## Attack Methodology: Targeted DeepFool & PGD Fallback

### 1. Targeted DeepFool
DeepFool is an iterative attack that finds the minimal perturbation to push an image across a decision boundary. For a *targeted* attack, we want to cross the boundary specifically into the target class. 
At each step, we approximate the decision boundary between the target class $t$ and all other classes $k$ as a hyperplane. We compute the minimal perturbation $r$ to reach this hyperplane:
$$ r = \frac{f_k - f_t}{\| \nabla f_t - \nabla f_k \|^2_2} (\nabla f_t - \nabla f_k) $$
We apply the smallest of these perturbations, add a slight overshoot to ensure crossing, and then project the image back into the allowed $\ell_2$ ball.

### 2. The PyTorch Leaf Variable Pitfall
During implementation, a common PyTorch autograd error was encountered:
`RuntimeError: you can only change requires_grad flags of leaf variables.`
This occurs because `x_adv` becomes a non-leaf variable in the computation graph after the first iteration. To fix this, we must call `x_adv = x_adv.detach().requires_grad_(True)` at the start of every iteration to treat it as a fresh input.

### 3. PGD Fallback Strategy
DeepFool relies on linear approximations of highly non-linear boundaries. Sometimes, the linear step is not aggressive enough to cross the boundary within a strict distance budget. If DeepFool fails to reach the target class, the script falls back to **Targeted Projected Gradient Descent (PGD)**. 
PGD explicitly maximizes the log-softmax output of the target class by computing the gradient of the target loss, taking a step in the *opposite* direction of the gradient (to minimize the loss), and projecting back into the $\ell_2$ ball.

### 4. The $\ell_2$ Quantization Buffer
Just like with $L_\infty$ in FGSM, 8-bit PNG quantization can cause tiny floating-point errors. To guarantee the server doesn't reject the submission with an `L2 too large` error, we define a safety buffer: `safe_thr = l2_thresh - 0.02` and project our perturbations into this slightly smaller ball.

## Exploitation Steps

1.  **Fetch Challenge & Weights**: Download the baseline `4` and the model `state_dict`.
2.  **Initialize Attack Loop**: Start with `x_adv = x` and set up the iterative loop.
3.  **Compute Gradients**: Use `torch.autograd.grad` to compute the Jacobian (gradients for all 10 classes) with respect to the input image.
4.  **DeepFool Update**: Calculate the minimal perturbation to push the image towards class `6`. Apply the perturbation with an `overshoot` of 1.02.
5.  **Projection**: Clip values to $[0,1]$ and project the perturbation back into the $\ell_2$ ball of radius `safe_thr`.
6.  **PGD Fallback**: If DeepFool doesn't reach class `6` after its iterations, switch to targeted PGD, stepping directly towards the target class gradient and projecting into the $\ell_2$ ball.
7.  **Submit**: Encode the final array to a base64 PNG and submit to the server.

## Proof of Concept (PoC) Script

```python
import os
import io
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BASE_URL = os.getenv("BASE_URL", "http://154.57.164.66:30805")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# ---------- Helpers ----------
def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    return np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel(), ord=2))

# ---------- Model ----------
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = (x01 - MNIST_MEAN) / MNIST_STD
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# ---------- Attack ----------
def targeted_attack(model, x, target, l2_thresh, max_iter=100):
    x_adv = x.clone()
    safe_thr = l2_thresh - 0.02
    overshoot = 1.02

    # Phase 1: Targeted DeepFool
    for i in range(max_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        log_probs = model(x_adv)[0]
        pred = torch.argmax(log_probs).item()
        
        if pred == target:
            print(f"[+] DeepFool reached target at iteration {i}")
            return x_adv.detach()
            
        jacobian = []
        for k in range(10):
            grad = torch.autograd.grad(log_probs[k], x_adv, retain_graph=True)[0]
            jacobian.append(grad[0, 0])
        jacobian = torch.stack(jacobian)
        
        f_t = log_probs[target]
        grad_t = jacobian[target]
        
        min_norm = float('inf')
        best_r = None
        
        for k in range(10):
            if k == target: continue
            f_k = log_probs[k]
            grad_k = jacobian[k]
            w = grad_t - grad_k
            dist = f_k - f_t
            w_norm_sq = torch.sum(w**2)
            if w_norm_sq > 1e-12:
                r = (dist / w_norm_sq) * w
                r_norm = torch.norm(r)
                if r_norm < min_norm:
                    min_norm = r_norm
                    best_r = r
                    
        if best_r is None or best_r.norm() == 0:
            best_r = 0.01 * grad_t
            
        x_adv = x_adv.detach() + (1 + overshoot) * best_r[None, None, ...]
        x_adv = torch.clamp(x_adv, 0, 1)
        diff = x_adv - x
        dist = torch.norm(diff, p=2)
        if dist > safe_thr:
            diff = diff * (safe_thr / dist)
            x_adv = x + diff

    # Phase 2: Targeted PGD Fallback
    print("[-] DeepFool didn't reach target. Falling back to targeted PGD...")
    for i in range(100):
        x_adv = x_adv.detach().requires_grad_(True)
        log_probs = model(x_adv)[0]
        pred = torch.argmax(log_probs).item()
        if pred == target:
            print(f"[+] PGD reached target at iteration {i}")
            return x_adv.detach()
            
        loss = -log_probs[target]
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad_norm = torch.norm(grad, p=2)
        if grad_norm > 0: grad = grad / grad_norm
            
        x_adv = x_adv.detach() - 0.05 * grad
        
        x_adv = torch.clamp(x_adv, 0, 1)
        diff = x_adv - x
        dist = torch.norm(diff, p=2)
        if dist > safe_thr:
            diff = diff * (safe_thr / dist)
            x_adv = x + diff

    return x_adv.detach()

# ---------- Main Execution ----------
if __name__ == "__main__":
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = x01_from_b64_png(ch["image_b64"])
    target, thr = int(ch["target"]), float(ch["l2_threshold"])

    open("deepfool_weights.pth", "wb").write(requests.get(f"{BASE_URL}/weights", timeout=10).content)
    model = SimpleClassifier().eval()
    model.load_state_dict(torch.load("deepfool_weights.pth", map_location="cpu"))

    x_tensor = torch.from_numpy(x[None, None, ...]).float()
    x_adv_tensor = targeted_attack(model, x_tensor, target, thr)
    x_adv_np = x_adv_tensor.numpy()[0, 0]
    
    r = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64_png_from_x01(x_adv_np)}, timeout=10)
    if r.status_code == 200:
        print(f"FLAG: {r.json().get('flag')}")
```

## Conclusion

While DeepFool is highly efficient for untargeted minimal perturbations, targeted DeepFool can struggle with the highly non-linear decision boundaries of deep CNNs. By implementing a robust pipeline that falls back to Targeted PGD, we guarantee crossing the boundary into the target class. Careful management of PyTorch's autograd leaf variables and strict $\ell_2$ projection with a quantization safety buffer ensured the adversarial image was accepted by the server validation checks.

### Flag
`HTB{d33pf00l_f00lz}`

--------------------------------

The error occurs because PyTorch tensors that are the result of operations are non-leaf variables, and you cannot call `.requires_grad = True` on them directly. We need to call `.detach().requires_grad_(True)` on `x_adv` at the start of each iteration to ensure it is treated as a fresh leaf variable for the autograd engine.

Additionally, I have added a robust targeted PGD fallback just in case the linear DeepFool step struggles to cross the highly non-linear boundary within the strict $L_2$ threshold, and refined the $L_2$ projection to be more precise.

Here is the corrected script:

```python
import os
import io
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BASE_URL = "http://154.57.164.66:30805"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# ---------- Helpers ----------
def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    return np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel(), ord=2))

# ---------- Model ----------
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = (x01 - MNIST_MEAN) / MNIST_STD
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# ---------- Targeted DeepFool Attack ----------
def targeted_deepfool(model, x, target, l2_thresh, max_iter=100):
    x_adv = x.clone()
    safe_thr = l2_thresh - 0.02  # Buffer for 8-bit PNG quantization
    overshoot = 1.02

    for i in range(max_iter):
        # FIX: detach and requires_grad_ at the start of the iteration
        x_adv = x_adv.detach().requires_grad_(True)
        
        log_probs = model(x_adv)[0]
        pred = torch.argmax(log_probs).item()
        
        if pred == target:
            print(f"[+] DeepFool reached target class {target} at iteration {i}")
            return x_adv.detach()
            
        # Compute gradients for all 10 classes
        jacobian = []
        for k in range(10):
            grad = torch.autograd.grad(log_probs[k], x_adv, retain_graph=True)[0]
            jacobian.append(grad[0, 0])
        jacobian = torch.stack(jacobian)
        
        f_t = log_probs[target]
        grad_t = jacobian[target]
        
        min_norm = float('inf')
        best_r = None
        
        # Find minimal perturbation to cross target boundary
        for k in range(10):
            if k == target:
                continue
            f_k = log_probs[k]
            grad_k = jacobian[k]
            
            w = grad_t - grad_k
            dist = f_k - f_t
            
            w_norm_sq = torch.sum(w**2)
            if w_norm_sq > 1e-12:
                r = (dist / w_norm_sq) * w
                r_norm = torch.norm(r)
                if r_norm < min_norm:
                    min_norm = r_norm
                    best_r = r
                    
        if best_r is None:
            best_r = 0.01 * grad_t
        elif best_r.norm() == 0:
            best_r = 0.01 * grad_t
            
        # Apply perturbation with overshoot
        x_adv = x_adv.detach() + (1 + overshoot) * best_r[None, None, ...]
        
        # Project to L2 ball and [0,1] box
        x_adv = torch.clamp(x_adv, 0, 1)
        diff = x_adv - x
        dist = torch.norm(diff, p=2)
        if dist > safe_thr:
            diff = diff * (safe_thr / dist)
            x_adv = x + diff

    # Fallback to Targeted PGD if DeepFool fails to cross the boundary
    print("[-] DeepFool didn't reach target. Falling back to targeted PGD...")
    for i in range(100):
        x_adv = x_adv.detach().requires_grad_(True)
        log_probs = model(x_adv)[0]
        pred = torch.argmax(log_probs).item()
        if pred == target:
            print(f"[+] PGD reached target class {target} at iteration {i}")
            return x_adv.detach()
            
        loss = -log_probs[target]
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        grad_norm = torch.norm(grad, p=2)
        if grad_norm > 0:
            grad = grad / grad_norm
            
        x_adv = x_adv.detach() - 0.05 * grad  # step opposite to loss gradient
        
        # Projection
        x_adv = torch.clamp(x_adv, 0, 1)
        diff = x_adv - x
        dist = torch.norm(diff, p=2)
        if dist > safe_thr:
            diff = diff * (safe_thr / dist)
            x_adv = x + diff

    return x_adv.detach()

# ---------- Main Execution ----------
if __name__ == "__main__":
    print(f"[*] Target: {BASE_URL}")
    
    # 1. Fetch challenge
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = x01_from_b64_png(ch["image_b64"])
    label = int(ch["label"])
    target = int(ch["target"])
    thr = float(ch["l2_threshold"])
    print(f"[*] Baseline label: {label}, Target: {target}, L2 Threshold: {thr}")

    # 2. Download and load weights
    print("[*] Downloading model weights...")
    open("deepfool_weights.pth", "wb").write(
        requests.get(f"{BASE_URL}/weights", timeout=10).content
    )
    
    model = SimpleClassifier().eval()
    model.load_state_dict(torch.load("deepfool_weights.pth", map_location="cpu"))

    x_tensor = torch.from_numpy(x[None, None, ...]).float()

    # 3. Run Attack
    print("[*] Computing perturbation...")
    x_adv_tensor = targeted_deepfool(model, x_tensor, target, thr)
    x_adv_np = x_adv_tensor.numpy()[0, 0]
    
    print(f"[*] Final L2 distance: {l2_dist(x, x_adv_np):.6f}")
    
    # 4. Check locally
    with torch.no_grad():
        pred_adv = int(torch.argmax(model(torch.from_numpy(x_adv_np[None, None]).float())).item())
    print(f"[*] Local adversarial prediction: {pred_adv}")

    # 5. Submit to server
    print("[*] Submitting to server...")
    r = requests.post(
        f"{BASE_URL}/submit",
        json={"image_b64": b64_png_from_x01(x_adv_np)},
        timeout=10
    )
    
    if r.status_code == 200:
        data = r.json()
        print("\n" + "="*40)
        print(f"FLAG: {data.get('flag')}")
        print("="*40)
    else:
        print(f"[-] Submission failed ({r.status_code}): {r.text}")
```

-----------------------------------
<img width="832" height="405" alt="image" src="https://github.com/user-attachments/assets/d62d304a-bf29-48e2-a732-69f86e01e354" />

