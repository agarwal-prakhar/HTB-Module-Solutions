Here is a detailed write-up of the ElasticNet Attack Challenge, documenting the debugging journey, why the initial attempts failed, and how the final FISTA optimization successfully extracted the flag.

***

# HackTheBox Academy: ElasticNet Attack Challenge Write-up

## Overview
The challenge required crafting an adversarial example to fool a simple MNIST digit classifier using the ElasticNet Attack (EAD). The EAD attack uses FISTA (Faster Iterative Shrinkage-Thresholding Algorithm) optimization with elastic-net regularization (combining L1 and L2 norms). The twist was that the adversarial perturbation had to simultaneously satisfy three distance constraints:
1. **Elastic-net distance:** `‖xadv−x‖2 + β‖xadv−x‖1 ≤ elastic_max`
2. **L2 cap:** `‖xadv−x‖2 ≤ l2_max`
3. **L1 cap:** `‖xadv−x‖1 ≤ l1_max`

To solve this, we had to implement the attack locally, download the model weights, and use binary search over the regularization constant `c` to find the perfect balance between misclassification and distance constraints.

---

## The Debugging Journey: Why It Failed

### Failure 1: The `.item()` Attribute Error
**The Error:** `AttributeError: 'int' object has no attribute 'item'`
**Why it happened:** The target label (`lab`) was fetched from the JSON API response as a standard Python integer. However, inside the `ead_attack` function, the code attempted to call `.item()` on it (e.g., `y.item()`), which is a method exclusive to PyTorch tensors. 
**The Fix:** Changed the comparisons inside the attack loop to use standard Python integer comparisons (e.g., `if pred.item() != y:` instead of `y.item()`).

### Failure 2: Insufficient Optimization Power (No Adversarial Found)
**The Error:** The script ran 20 iterations of binary search but never found a misclassified image. 
**Why it happened:** The initial hyperparameters were too weak. The learning rate (`lr=0.01`) was too small, the maximum iterations (`max_iter=500`) were too short, and the upper bound for the binary search (`c_high=10.0`) was too low for this specific model's confidence. The optimizer simply wasn't taking large enough steps to cross the decision boundary.
**The Fix:** Increased `lr` to `0.1`, `max_iter` to `1000`, and expanded the binary search range up to `c_high=100.0`.

### Failure 3: The Zero Gradient Sign Error (The Core Bug)
**The Error:** Even with better hyperparameters, the attack still failed to find an adversarial example.
**Why it happened:** This was a critical mathematical bug in the Carlini-Wagner (CW) loss function. The original code computed the loss as:
`loss = torch.clamp(max_other - target_logit, min=0.0)`
When the model is highly confident in the baseline image (which it was for the digit '9'), the `target_logit` is much larger than `max_other`. Therefore, `max_other - target_logit` is a negative number. The `torch.clamp(..., min=0.0)` immediately clipped this negative value to `0.0`. A loss of `0.0` produces a gradient of `0.0`. With no gradient, the FISTA optimizer stood completely still and never modified the image.
**The Fix:** Flipped the loss formula to actively push the true label's logit *down* and the other logits *up*:
`loss = torch.clamp(target_logit - max_other, min=0.0)`
Now, when the model is confident, the loss is positive, generating a strong gradient to push the image away from the true label.

### Failure 4: The Strict Bounds Check (`ValueError: Input must be in [0,1]`)
**The Error:** Once the gradients started flowing, the script crashed with `ValueError: Input must be in [0,1]`.
**Why it happened:** FISTA optimization uses a momentum step (`y_k = z_k + momentum`). Because of this momentum, the intermediate variable `y_k` can temporarily overshoot and go outside the valid pixel range of `[0, 1]`. The helper function `mnist_normalize` had a strict check that raised an error if the input was outside `[0, 1]`, killing the script during the intermediate steps.
**The Fix:** Bypassed the strict `mnist_normalize` helper function *inside* the attack loop. Instead, the normalization was inlined directly: `norm_y = (y_k - MNIST_MEAN) / MNIST_STD`. The actual candidate image (`z_k`) was still strictly clipped to `[0, 1]` using `torch.clamp(z_k, 0.0, 1.0)`, ensuring the final output remained valid.

---

## Why It Worked: The Final Execution

With all bugs fixed, the FISTA algorithm functioned exactly as intended. 

1. **Binary Search:** The script tested `c=50.0`. The attack found an adversarial example, but the L2 and Elastic distances were too high (violating the constraints). 
2. **Convergence:** The binary search automatically reduced `c` to prioritize minimizing the distortion over misclassification. It tested `c=25.0`, then `c=12.5`, and finally `c=6.25`.
3. **The Sweet Spot:** At `c=6.25`, the FISTA optimizer found an adversarial image that caused the model to misclassify the '9' as an '8', while satisfying all three distance bounds:
   * **Elastic distance:** 1.19 (Max allowed: 1.8) ✅
   * **L2 distance:** 1.08 (Max allowed: 1.4) ✅
   * **L1 distance:** 10.63 (Max allowed: 80.0) ✅

The script encoded the valid image into a base64 PNG, submitted it to the `/submit` endpoint, and the server validated the results and returned the flag.

## Flag
**`HTB{3l4st1cn3t_3l4st1c1s1ng_3l4st1c}`**

-----------------------------
```
import os, io, base64, numpy as np, requests, torch, torch.nn as nn
from PIL import Image

# --- Configuration ---
BASE_URL = os.getenv("BASE_URL", "http://154.57.164.82:32382")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# --- API Helpers ---
def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def compute_distances(a: np.ndarray, b: np.ndarray, beta: float) -> dict:
    diff = a - b
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff**2)))
    linf = float(np.max(np.abs(diff)))
    elastic = l2 + beta * l1
    return {"l1": l1, "l2": l2, "linf": linf, "elastic": elastic}

# --- Model Architecture ---
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    if x01.ndim != 4 or x01.shape[1] != 1:
        raise ValueError(f"Expected (N,1,28,28), got {x01.shape}")
    if x01.min() < 0.0 or x01.max() > 1.0:
        raise ValueError("Input must be in [0,1]")
    return (x01 - MNIST_MEAN) / MNIST_STD

# --- EAD FISTA Attack ---
def ead_attack(x, y, beta, c, lr=0.1, max_iter=1000):
    z_prev = x.clone()
    y_k = x.clone()
    t_k = 1.0
    best_adv = None
    best_dist = float('inf')
    
    for i in range(max_iter):
        y_k.requires_grad = True
        # Normalize directly to avoid the strict bounds check on intermediate y_k
        norm_y = (y_k - MNIST_MEAN) / MNIST_STD
        logits = model(norm_y)
        
        target_logit = logits[0, y]
        mask = torch.eye(10, device=x.device)[y].bool().view(1, 10)
        other_logits = logits.masked_fill(mask, -float('inf'))
        max_other = torch.max(other_logits, dim=1)[0]
        
        # CW loss: Push target logit DOWN, other logits UP
        loss = torch.clamp(target_logit - max_other, min=0.0)
        
        total_loss = c * loss + torch.norm(y_k - x, p=2)**2 / 2.0
        grad = torch.autograd.grad(total_loss, y_k)[0]
        
        with torch.no_grad():
            x_k = y_k - lr * grad
            
            shrink = torch.sign(x_k - x) * torch.relu(torch.abs(x_k - x) - lr * beta)
            z_k = x + shrink
            z_k = torch.clamp(z_k, 0.0, 1.0)  # Keep the actual candidate in [0, 1]
            
            t_next = (1 + (1 + 4 * t_k**2)**0.5) / 2.0
            y_next = z_k + ((t_k - 1) / t_next) * (z_k - z_prev)
            
            z_prev = z_k
            t_k = t_next
            y_k = y_next
            
            # Check if misclassified using the clamped z_k
            pred = torch.argmax(model((z_k - MNIST_MEAN) / MNIST_STD), dim=1)
            if pred.item() != y:
                dist = compute_distances(z_k.cpu().numpy()[0,0], x.cpu().numpy()[0,0], beta)
                if dist['elastic'] < best_dist:
                    best_dist = dist['elastic']
                    best_adv = z_k.clone()
                    if dist['elastic'] <= elastic_max and dist['l2'] <= l2_max and dist['l1'] <= l1_max:
                        return best_adv, True
    return best_adv, False

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[*] Fetching challenge from {BASE_URL}...")
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x_orig_np = x01_from_b64_png(ch["image_b64"])
    lab = int(ch["label"])
    beta = float(ch["beta"])
    elastic_max = float(ch["elastic_max"])
    l2_max = float(ch["l2_max"])
    l1_max = float(ch["l1_max"])

    print(f"[*] Label: {lab}, Beta: {beta}, Elastic Max: {elastic_max}, L2 Max: {l2_max}, L1 Max: {l1_max}")

    print("[*] Downloading model weights...")
    wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
    open("elasticnet_weights.pth", "wb").write(wt)

    model = SimpleClassifier().eval()
    state = torch.load("elasticnet_weights.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)

    x_tensor = torch.from_numpy(x_orig_np[None, None, ...]).float()

    print("[*] Starting binary search for c...")
    c_low = 0.0
    c_high = 100.0
    best_valid_adv = None

    for iteration in range(20):
        c = (c_low + c_high) / 2.0
        print(f"\n[-] Iteration {iteration+1}: Testing c={c:.6f} (Range: {c_low:.6f} - {c_high:.6f})")
        
        adv, success = ead_attack(x_tensor, lab, beta, c, lr=0.1, max_iter=1000)
        
        if adv is not None:
            dist = compute_distances(adv.cpu().numpy()[0,0], x_orig_np, beta)
            print(f"  [+] Adversarial example found! Distances -> Elastic: {dist['elastic']:.4f}, L2: {dist['l2']:.4f}, L1: {dist['l1']:.4f}")
            
            if dist['elastic'] <= elastic_max and dist['l2'] <= l2_max and dist['l1'] <= l1_max:
                print("  [**] SUCCESS! All constraints satisfied.")
                best_valid_adv = adv
                break
            else:
                print("  [-] Constraints violated. Reducing c to minimize distortion.")
                c_high = c
        else:
            print("  [-] No adversarial example found. Increasing c to prioritize misclassification.")
            c_low = c

    if best_valid_adv is not None:
        print("\n[*] Submitting valid adversarial example...")
        adv_np = best_valid_adv.cpu().numpy()[0,0]
        b64 = b64_png_from_x01(adv_np)
        
        res = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64}, timeout=10).json()
        print("\n--- Submission Results ---")
        print(res)
        if res.get("ok"):
            print(f"\n[+] FLAG: {res.get('flag')}")
    else:
        print("\n[!] Failed to find a valid adversarial example within the constraints.")

```
