Here is a detailed write-up of the Jacobian-based Saliency Map Attack (JSMA) Challenge, documenting the methodology, the debugging process, and why the final attack vector successfully bypassed the classifier.

***

# HackTheBox Academy: Jacobian-based Saliency Map Attack (JSMA) Write-up

## Overview
The challenge required crafting a **targeted adversarial example** to fool a LeNet-5 style MNIST digit classifier. Unlike untargeted attacks (which just aim for *any* misclassification), a targeted attack forces the model to predict a specific class. 

The constraints were strict:
1.  **Targeted Misclassification:** The model must classify the adversarial image as the `target_class` (e.g., `7`).
2.  **L0 Budget:** At most `50` pixels could be modified out of the 784 total pixels.
3.  **L2 Constraint:** The overall L2 distance could not exceed `8.0`.
4.  **Valid Pixel Space:** All pixels had to remain in the `[0, 1]` range.

## Understanding JSMA
JSMA is an iterative, greedy attack algorithm that modifies one pixel at a time. At each step, it computes the **Jacobian matrix** (the gradient of every output class with respect to every input pixel). It uses this matrix to build a **Saliency Map**, which scores how useful each pixel is for increasing the target class score while decreasing the scores of all other classes. The algorithm picks the highest-scoring pixel and modifies it, repeating the process until the target class is achieved or the budget is exhausted.

---

## The Debugging Journey: Why It Failed

### Failure 1: The Single-Modification Restriction & Small Step Size
**The Error:** The initial script exhausted the 50-pixel L0 budget but failed to change the prediction.
**Why it happened:** The initial code restricted each pixel to be modified exactly *once*, and used a very small step size (`theta = 0.1`). Because JSMA relies on iteratively pushing pixels, modifying 50 distinct pixels by a tiny `0.1` increment was mathematically insufficient to cross the model's decision boundary.
**The Fix:** Allowed pixels to be modified *multiple times* (until they hit the `[0, 1]` saturation bounds) and increased the step size.

### Failure 2: The Vanishing Gradient (Softmax vs. Logits)
**The Error:** Even with multiple modifications and a larger step size, the attack stalled. The logs showed it stuck at `Pred=1, Target=7, L0=38, L2=2.02` for thousands of iterations. The saliency map was registering zero, meaning no pixels were deemed useful to modify.
**Why it happened:** This was the core mathematical bug. The model's `forward` pass ended with `F.log_softmax(x, dim=1)`. The Jacobian was being computed on these **softmax probabilities**. Softmax functions squash logits into a probability distribution `[0, 1]`. As the model becomes highly confident in its prediction (even if it's the wrong one), the softmax gradient approaches zero. This "vanishing gradient" meant the Jacobian matrix became essentially zero, blinding the JSMA algorithm and causing it to stall completely.
**The Fix:** Removed `F.log_softmax` from the `forward` pass and instead computed the Jacobian on the **raw logits** (the pre-softmax linear outputs). Because logits are unbounded, their gradients do not vanish, allowing the attack to maintain a clear signal toward the target class regardless of the model's current confidence.

### Failure 3: Insufficient Step Size for Rapid Convergence
**The Error:** With the logits fixed, the attack still took too many iterations to reach the target, risking timeouts or bouncing around the decision boundary.
**Why it happened:** A small `theta` (like `0.1`) means the algorithm takes tiny steps toward the boundary, requiring hundreds of modifications.
**The Fix:** Increased the step size `theta` to `1.0`. In JSMA, using a step size of `1.0` means that when a pixel is identified as salient, it is immediately pushed to its maximum (`1.0`) or minimum (`0.0`) bound. This is highly effective for L0 attacks, as it extracts the maximum possible influence out of every single pixel modified.

---

## Why It Worked: The Final Execution

With the Jacobian correctly computing on raw logits and an aggressive step size of `1.0`, the JSMA algorithm functioned perfectly.

1.  **Iteration 0:** The clean image of a `1` was fed in. Prediction = `1`.
2.  **Iterations 1-7:** The algorithm computed the logits' Jacobian, found the specific pixels that most strongly increased the "7" logit while decreasing the "1" logit, and aggressively snapped them to `1.0` or `0.0`.
3.  **Iteration 7:** The model's decision boundary was crossed. The classifier's logits shifted enough that `7` became the highest-scoring class.

The attack achieved its goal by modifying **only 7 pixels** (well under the 50-pixel budget) with an L2 distance of **2.6458** (well under the 8.0 max L2 constraint).

The script encoded the image, submitted it to the `/submit` API, and the server validated that all constraints were met, returning the flag.

## Flag
**`HTB{j5m4_l4b_l3n3t_l3c00l}`**

--------------

```
import os, io, base64, numpy as np, requests, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

# --- Configuration ---
# Replace with your actual instance URL
BASE_URL = os.getenv("BASE_URL", "http://154.57.164.82:30308")
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

# --- Model Architecture ---
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x  # Return raw logits (pre-softmax) for Jacobian!

def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    return (x01 - MNIST_MEAN) / MNIST_STD

def compute_jacobian(model, x):
    x.requires_grad_(True)
    logits = model(mnist_normalize(x))
    jacobian = torch.zeros(10, 28, 28)
    for i in range(10):
        if x.grad is not None:
            x.grad.zero_()
        logits[0, i].backward(retain_graph=True)
        jacobian[i] = x.grad[0, 0].clone()
    x.requires_grad_(False)
    return jacobian

# --- JSMA Attack ---
def jsma_attack(model, x_orig, target, budget, max_l2, theta=1.0, max_iter=5000):
    x_adv = x_orig.clone()
    
    for iteration in range(max_iter):
        logits = model(mnist_normalize(x_adv))
        pred = torch.argmax(logits, dim=1).item()
        
        l0_dist = (torch.abs(x_adv - x_orig) > 1e-6).sum().item()
        l2_dist = torch.norm(x_adv - x_orig).item()
        
        if iteration % 10 == 0:
            print(f"  Iter {iteration}: Pred={pred}, Target={target}, L0={l0_dist}, L2={l2_dist:.2f}")
        
        if pred == target:
            print(f"  [+] Target reached! Iter: {iteration}, L0: {l0_dist}, L2: {l2_dist:.4f}")
            return x_adv
            
        if l0_dist >= budget:
            print(f"  [-] L0 budget exhausted! L0: {l0_dist}, L2: {l2_dist:.4f}")
            break
            
        jacobian = compute_jacobian(model, x_adv)
        
        grad_target = jacobian[target]
        mask = torch.ones(10, dtype=torch.bool)
        mask[target] = False
        grad_others = jacobian[mask].sum(dim=0)
        
        # Saliency for increasing pixel
        saliency_inc = torch.zeros_like(grad_target)
        cond_inc = (grad_target > 0) & (grad_others < 0)
        saliency_inc[cond_inc] = grad_target[cond_inc] * (-grad_others[cond_inc])
        saliency_inc[x_adv[0, 0] >= 1.0 - 1e-6] = -1  # mask saturated pixels
        
        # Saliency for decreasing pixel
        saliency_dec = torch.zeros_like(grad_target)
        cond_dec = (grad_target < 0) & (grad_others > 0)
        saliency_dec[cond_dec] = (-grad_target[cond_dec]) * grad_others[cond_dec]
        saliency_dec[x_adv[0, 0] <= 0.0 + 1e-6] = -1  # mask saturated pixels
        
        max_inc = torch.max(saliency_inc).item()
        max_dec = torch.max(saliency_dec).item()
        
        if max_inc == 0 and max_dec == 0:
            print("  [-] No salient pixel found (all gradients unfavorable).")
            break
            
        if max_inc >= max_dec and max_inc > 0:
            max_idx = torch.argmax(saliency_inc).item()
            r, c = max_idx // 28, max_idx % 28
            x_adv[0, 0, r, c] = torch.clamp(x_adv[0, 0, r, c] + theta, 0.0, 1.0)
        elif max_dec > 0:
            max_idx = torch.argmax(saliency_dec).item()
            r, c = max_idx // 28, max_idx % 28
            x_adv[0, 0, r, c] = torch.clamp(x_adv[0, 0, r, c] - theta, 0.0, 1.0)
            
    return None

# --- Main Execution ---
if __name__ == "__main__":
    print(f"[*] Fetching challenge from {BASE_URL}...")
    ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
    x_orig_np = x01_from_b64_png(ch["image_b64"])
    orig_label = int(ch["original_label"])
    target = int(ch["target_class"])
    budget = int(ch["l0_budget"])
    max_l2 = float(ch["max_l2"])

    print(f"[*] Original Label: {orig_label}, Target: {target}, L0 Budget: {budget}, Max L2: {max_l2}")

    print("[*] Downloading model weights...")
    wt = requests.get(f"{BASE_URL}/weights", timeout=10).content
    open("jsma_weights.pth", "wb").write(wt)

    model = MNISTClassifier().eval()
    state = torch.load("jsma_weights.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)

    x_tensor = torch.from_numpy(x_orig_np[None, None, ...]).float()

    print("[*] Starting JSMA attack...")
    adv_tensor = jsma_attack(model, x_tensor, target, budget, max_l2, theta=1.0, max_iter=5000)

    if adv_tensor is not None:
        print("\n[*] Submitting valid adversarial example...")
        adv_np = adv_tensor.cpu().numpy()[0,0]
        b64 = b64_png_from_x01(adv_np)
        
        res = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64}, timeout=10).json()
        print("\n--- Submission Results ---")
        print(res)
        if res.get("success"):
            print(f"\n[+] FLAG: {res.get('flag')}")
    else:
        print("\n[!] Failed to find a valid adversarial example within the constraints.")
```
