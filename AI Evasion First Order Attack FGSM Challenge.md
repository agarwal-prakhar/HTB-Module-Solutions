# Lab Write-Up: FGSM Challenge

## Overview
This challenge introduces adversarial machine learning, specifically focusing on the **Fast Gradient Sign Method (FGSM)**. The objective is to perturb a baseline MNIST digit image just enough to cause a classifier to misclassify it, while keeping the perturbation strictly within an $L_\infty$ budget of $\epsilon = 0.25$.

## Reconnaissance & API Exploration

The challenge provides a REST API to interact with the target classifier. By inspecturing the endpoints, we can map out our attack strategy:

1.  **`GET /challenge`**: Returns the baseline MNIST image, the ground truth label (`label=1`), and the strict perturbation budget (`epsilon=0.25`).
2.  **`GET /weights`**: Returns the PyTorch `state_dict` for the model. This is a critical flaw for the defender: providing the weights turns this into a **white-box attack**, allowing us to compute gradients locally.
3.  **`POST /predict`**: Allows us to test arbitrary images against the model's inference pipeline.
4.  **`POST /submit`**: The validation endpoint. It checks if the $L_\infty$ distance between our image and the baseline is $\le \epsilon$ and if the model's prediction differs from the ground truth.

The model architecture (`SimpleClassifier`) is also provided. It takes a $28\times28$ image in $[0,1]$ space, normalizes it internally, passes it through two convolutional layers, and outputs log-softmax probabilities.

## The Attack: Fast Gradient Sign Method (FGSM)

FGSM is a white-box adversarial attack that leverages the gradients of the loss function with respect to the input image. Instead of optimizing the *weights* to minimize the loss (training), we optimize the *input* to **maximize** the loss, pushing the image out of its true classification boundary.

The formula for untargeted FGSM is:
$$ x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(x, y_{true})) $$

Where:
*   $x$ is the original image.
*   $\epsilon$ is the perturbation budget (`0.25`).
*   $\text{sign}$ extracts the direction of the gradient (-1, 0, or 1).
*   $J(x, y_{true})$ is the loss (Negative Log-Likelihood) evaluated against the true label.

## The Pitfall: PNG Quantization & The $L_\infty$ Budget

An initial, naive implementation of FGSM computes the adversarial image in floating-point space and saves it as a PNG. However, PNGs use 8-bit integers (0-255). When we save an image, a pixel value of `0.25` is mapped to `0.25 * 255 = 63.75`. 

Due to rounding, `63.75` becomes `64`. When the server decodes this PNG, it reads `64` and converts it back to $[0,1]$ space: `64 / 255 = 0.25098`. 

This quantization error causes the $L_\infty$ distance to become `0.25098`, which strictly violates the server's `0.250000` budget, resulting in a `400 Bad Request`.

### The Fix: Integer-Space Perturbation
To bypass this, the attack must be computed and bounded in 8-bit integer space. 
1. We calculate the maximum integer perturbation allowed: `int(0.25 * 255) = 63`.
2. We round our FGSM perturbation to integers.
3. We clip the integer difference between the adversarial image and the original image so it never exceeds `63`.

This guarantees that when the server converts our PNG back to $[0,1]$ space, the maximum distance will be `63 / 255 = 0.247058...`, safely under the `0.25` budget.

## Exploitation Steps

1.  **Fetch the Challenge**: Download the baseline image and label.
2.  **Load the White-Box Model**: Download the model weights and load them into the locally reconstructed `SimpleClassifier` class.
3.  **Compute Gradients**: Pass the image through the model with `requires_grad_(True)`. Calculate the NLL loss against the true label (`1`) and call `.backward()` to compute input gradients.
4.  **Generate Perturbation**: Take the `sign()` of the gradients, multiply by $\epsilon$, and add to the original image.
5.  **Integer Quantization**: Convert to 8-bit integers and strictly enforce the `max_diff = 63` constraint.
6.  **Submit**: Encode the array to a base64 PNG and send it to the `/submit` endpoint.

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

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# ---------- Helpers ----------
def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    return np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)

def b64_png_from_int(x_int: np.ndarray) -> str:
    x255 = np.clip(x_int, 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

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

# ---------- Execution ----------
if __name__ == "__main__":
    # 1. Fetch challenge
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = x01_from_b64_png(ch["image_b64"])
    label = int(ch["label"])
    eps = float(ch["epsilon"])

    # 2. Download and load weights
    open("fgsm_weights.pth", "wb").write(
        requests.get(f"{BASE_URL}/weights", timeout=10).content
    )
    model = SimpleClassifier().eval()
    model.load_state_dict(torch.load("fgsm_weights.pth", map_location="cpu"))

    # 3. Compute FGSM perturbation
    xt = torch.from_numpy(x[None, None, ...]).float()
    xt.requires_grad_(True)
    
    log_probs = model(xt)
    loss = torch.nn.functional.nll_loss(log_probs, torch.tensor([label]))
    model.zero_grad()
    loss.backward()
    
    grad_sign = xt.grad.sign().detach().numpy()[0, 0]
    x_adv = np.clip(x + eps * grad_sign, 0.0, 1.0)
    
    # 4. Strict L_inf quantization in 8-bit integer space
    x_int = (x * 255.0).round().astype(np.int16)
    x_adv_int = (x_adv * 255.0).round().astype(np.int16)
    max_diff = int(eps * 255.0 + 1e-6) # 0.25 * 255 = 63.75 -> 63
    
    diff = np.clip(x_adv_int - x_int, -max_diff, max_diff)
    x_adv_int_clipped = np.clip(x_int + diff, 0, 255).astype(np.uint8)
    x_adv_final = x_adv_int_clipped.astype(np.float32) / 255.0

    # 5. Submit to server
    r = requests.post(
        f"{BASE_URL}/submit",
        json={"image_b64": b64_png_from_int(x_adv_int_clipped)},
        timeout=10
    )
    
    if r.status_code == 200:
        data = r.json()
        print(f"FLAG: {data.get('flag')}")
    else:
        print(f"Failed ({r.status_code}): {r.text}")
```

## Conclusion

By exploiting the white-box nature of the challenge, we successfully computed the gradient of the loss function with respect to the input image. After applying the FGSM perturbation and carefully handling 8-bit PNG quantization limits to stay strictly within the $L_\infty$ budget, the classifier was misled into predicting an `8` instead of a `1`.

### Flag
`HTB{f457_gr4d13n7_516n_m15l34d5_mn157}`

----------------------------

The issue is caused by 8-bit PNG quantization. 

When you add `0.25` to a pixel with a value of `0`, you get `0.25`. 
To save this as a PNG, the script multiplies by `255` and rounds: `0.25 * 255 = 63.75`. 
The `round()` function rounds `63.75` to `64`. 
When the server decodes your PNG, it reads `64` and converts it back to `[0,1]` space: `64 / 255 = 0.25098`, which is strictly greater than your `0.25` budget, causing the server to reject it.

To fix this, we need to do the perturbation and quantization in 8-bit integer space, ensuring the integer difference between the original image and the adversarial image never exceeds `floor(0.25 * 255) = 63`.

Here is the corrected `solve.py`:

```python
import os
import io
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# ---------- Helpers ----------
def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    return np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)

def b64_png_from_int(x_int: np.ndarray) -> str:
    """Convert 8-bit integer array to base64 PNG safely."""
    x255 = np.clip(x_int, 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))

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

# ---------- Main Execution ----------
if __name__ == "__main__":
    print(f"[*] Target: {BASE_URL}")
    
    # 1. Fetch challenge
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x = x01_from_b64_png(ch["image_b64"])
    label = int(ch["label"])
    eps = float(ch["epsilon"])
    print(f"[*] Baseline label: {label}, Epsilon: {eps}")

    # 2. Download and load weights
    print("[*] Downloading model weights...")
    open("fgsm_weights.pth", "wb").write(
        requests.get(f"{BASE_URL}/weights", timeout=10).content
    )
    
    model = SimpleClassifier().eval()
    model.load_state_dict(torch.load("fgsm_weights.pth", map_location="cpu"))

    # 3. Compute FGSM perturbation
    print("[*] Computing FGSM perturbation...")
    xt = torch.from_numpy(x[None, None, ...]).float()
    xt.requires_grad_(True)
    
    log_probs = model(xt)
    loss = torch.nn.functional.nll_loss(log_probs, torch.tensor([label]))
    model.zero_grad()
    loss.backward()
    
    grad_sign = xt.grad.sign().detach().numpy()[0, 0]
    x_adv = np.clip(x + eps * grad_sign, 0.0, 1.0)
    
    # 4. FIX: Quantize to 8-bit integers and strictly enforce L_inf in integer space
    # The original image from the server is already quantized to 8-bit:
    x_int = (x * 255.0).round().astype(np.int16)
    x_adv_int = (x_adv * 255.0).round().astype(np.int16)
    
    # Calculate the maximum integer distance allowed
    # e.g., 0.25 * 255 = 63.75 -> max_diff = 63
    max_diff = int(eps * 255.0 + 1e-6)
    
    # Clip the difference so PNG quantization cannot push us over the budget
    diff = np.clip(x_adv_int - x_int, -max_diff, max_diff)
    x_adv_int_clipped = np.clip(x_int + diff, 0, 255).astype(np.uint8)
    
    # Convert back to [0,1] for local verification
    x_adv_final = x_adv_int_clipped.astype(np.float32) / 255.0
    
    print(f"[*] L_inf distance (float): {linf(x, x_adv_final):.6f}")
    print(f"[*] L_inf distance (int):   {np.max(np.abs(x_adv_int_clipped.astype(np.int16) - x_int))}")

    # 5. Check locally
    with torch.no_grad():
        pred_adv = int(torch.argmax(model(torch.from_numpy(x_adv_final[None, None]).float())).item())
    print(f"[*] Local adversarial prediction: {pred_adv}")

    # 6. Submit to server
    print("[*] Submitting to server...")
    r = requests.post(
        f"{BASE_URL}/submit",
        json={"image_b64": b64_png_from_int(x_adv_int_clipped)},
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

Run it with:
```bash
python3 solve.py
```
