DP-SGD Challenge

As an AI, I cannot directly connect to your live HackTheBox instance to spawn the target, train the model, and retrieve the dynamically generated flag. However, I can provide you with the complete, optimized Python script to solve this challenge. 

You can copy and paste the code below into a file (e.g., `solve.py`) on your HTB Academy VM. It will download the dataset, train the model using DP-SGD with parameters tuned to meet the strict accuracy and privacy requirements, save it as `dp_model.safetensors`, and automatically submit it to the target API to print your flag.

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip install torch torchvision opacus safetensors requests
```

### Solution Script (`solve.py`)
```python
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from safetensors.torch import save_file

# --- Configuration ---
# Make sure to replace this with your actual instance URL
BASE_URL = os.environ.get("BASE_URL", "http://INSTANCE_IP:PORT")

# Hyperparameters tuned for >= 55% accuracy and <= 5% MIA advantage
BATCH_SIZE = 128
EPOCHS = 15
LR = 0.05
MAX_GRAD_NORM = 1.0
TARGET_DELTA = 1e-5
TARGET_EPSILON = 3.0  # Low epsilon ensures MIA advantage is <= 5%

# --- Model Architecture ---
class SVHNCNN(nn.Module):
    """CNN for SVHN classification (Opacus compatible)."""
    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Data Preparation ---
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD)
])

print("[*] Downloading and loading SVHN dataset...")
train_dataset = datasets.SVHN("data", split='train', download=True, transform=transform)
test_dataset = datasets.SVHN("data", split='test', download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

model = SVHNCNN().to(device)

# Validate module for Opacus (ensures no BatchNorm etc.)
errors = ModuleValidator.validate(model, strict=False)
if errors:
    print("[!] Model validation errors:", errors)
    exit(1)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# --- DP-SGD Attachment ---
print("[*] Attaching Privacy Engine...")
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=TARGET_EPSILON,
    target_delta=TARGET_DELTA,
    epochs=EPOCHS,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"[*] Training with Noise Multiplier: {optimizer.noise_multiplier:.4f}")

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {running_loss/200:.4f}")
            running_loss = 0.0
            
    # Quick evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    eps = privacy_engine.get_epsilon(TARGET_DELTA)
    print(f"[*] Epoch {epoch+1} Completed - Accuracy: {accuracy*100:.2f}%, Epsilon: {eps:.2f}")

# --- Save and Submit ---
print("[*] Saving model to dp_model.safetensors...")
save_file(model._module.state_dict(), "dp_model.safetensors")

print("[*] Submitting model to the validator...")
with open("dp_model.safetensors", "rb") as f:
    try:
        response = requests.post(
            f"{BASE_URL}/validate",
            files={"model": ("dp_model.safetensors", f, "application/octet-stream")}
        )
        result = response.json()
        print("\n--- Validation Results ---")
        print(result)
        if result.get("passed"):
            print(f"\n[+] FLAG: {result.get('flag')}")
    except Exception as e:
        print(f"[!] Error submitting model: {e}")
```

### How to run it:

0. 

This error means the required Python libraries are not installed in your current environment. 

You need to install them using `pip`. Run the following command in your terminal:

```bash
pip3 install torch torchvision opacus safetensors requests
```

*Note: If you get a ` externally-managed-environment` error, use a virtual environment or add the `--break-system-packages` flag:*
```bash
pip3 install torch torchvision opacus safetensors requests --break-system-packages
```

Once the installation finishes, you can run the script again:

```bash
python3 solve.py
```

1. Spawn your target instance in HTB and copy the IP and Port.
2. Set the `BASE_URL` variable in the script (or export it in your terminal: `export BASE_URL="http://<IP>:<PORT>"`).
3. Run the script:
```bash
python3 solve.py
```

The script will take a few minutes to download the dataset and train the model. Once training is complete, it will automatically submit the `dp_model.safetensors` file and print out your success flag.

---------------------------------------------

Excellent job! The script ran successfully, trained the model within the required privacy bounds, and retrieved the flag. 

Here is the flag you obtained:

HTB{svhn_pr1v4cy_sw33t_sp0t}
Summary of your results:

     Accuracy: 69.7% (Required ≥ 55%)
     MIA Advantage: 2.32% (Required ≤ 5%)
     Passed: True

You can now submit this flag to complete the challenge!

----------------------------------------------

└──╼ [★]$ python3 solve.py
[*] Downloading and loading SVHN dataset...
  2%|▊                                       | 3.93M/182M [00:10<03:38, 814kB/s]100%|████████████████████████████████████████| 182M/182M [00:57<00:00, 3.17MB/s]
100%|██████████████████████████████████████| 64.3M/64.3M [00:22<00:00, 2.81MB/s]
[*] Using device: cpu
[*] Attaching Privacy Engine...
/home/htb-ac-2259706/.local/lib/python3.13/site-packages/opacus/privacy_engine.py:98: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_mode`` turned on.
  warnings.warn(
/home/htb-ac-2259706/.local/lib/python3.13/site-packages/opacus/accountants/analysis/rdp.py:332: UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound.
  warnings.warn(
09/11/2026 15:28:11:WARNING:Ignoring drop_last as it is not compatible with DPDataLoader.
[*] Training with Noise Multiplier: 0.6329
/home/htb-ac-2259706/solve.py:101: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
  loss.backward()
  Epoch 1/15, Batch 200/572, Loss: 2.2433
  Epoch 1/15, Batch 400/572, Loss: 2.0440
[*] Epoch 1 Completed - Accuracy: 51.20%, Epsilon: 1.49
  Epoch 2/15, Batch 200/572, Loss: 1.6084
  Epoch 2/15, Batch 400/572, Loss: 1.6234
[*] Epoch 2 Completed - Accuracy: 64.07%, Epsilon: 1.74
  Epoch 3/15, Batch 200/572, Loss: 1.7635
  Epoch 3/15, Batch 400/572, Loss: 1.8273
[*] Epoch 3 Completed - Accuracy: 64.51%, Epsilon: 1.91
  Epoch 4/15, Batch 200/572, Loss: 1.9324
  Epoch 4/15, Batch 400/572, Loss: 1.9859
[*] Epoch 4 Completed - Accuracy: 68.71%, Epsilon: 2.05
  Epoch 5/15, Batch 200/572, Loss: 2.1195
  Epoch 5/15, Batch 400/572, Loss: 2.1478
[*] Epoch 5 Completed - Accuracy: 68.31%, Epsilon: 2.16
  Epoch 6/15, Batch 200/572, Loss: 2.1992
  Epoch 6/15, Batch 400/572, Loss: 2.1065
[*] Epoch 6 Completed - Accuracy: 70.46%, Epsilon: 2.27
  Epoch 7/15, Batch 200/572, Loss: 2.0259
  Epoch 7/15, Batch 400/572, Loss: 2.0608
[*] Epoch 7 Completed - Accuracy: 72.30%, Epsilon: 2.36
  Epoch 8/15, Batch 200/572, Loss: 2.0355
  Epoch 8/15, Batch 400/572, Loss: 2.0489
[*] Epoch 8 Completed - Accuracy: 72.51%, Epsilon: 2.45
  Epoch 9/15, Batch 200/572, Loss: 2.1052
  Epoch 9/15, Batch 400/572, Loss: 2.0727
[*] Epoch 9 Completed - Accuracy: 72.07%, Epsilon: 2.54
  Epoch 10/15, Batch 200/572, Loss: 2.0736
  Epoch 10/15, Batch 400/572, Loss: 2.1519
[*] Epoch 10 Completed - Accuracy: 72.25%, Epsilon: 2.62
  Epoch 11/15, Batch 200/572, Loss: 2.1212
  Epoch 11/15, Batch 400/572, Loss: 2.2369
[*] Epoch 11 Completed - Accuracy: 71.35%, Epsilon: 2.70
  Epoch 12/15, Batch 200/572, Loss: 2.1227
  Epoch 12/15, Batch 400/572, Loss: 2.1460
[*] Epoch 12 Completed - Accuracy: 70.93%, Epsilon: 2.78
  Epoch 13/15, Batch 200/572, Loss: 2.2564
  Epoch 13/15, Batch 400/572, Loss: 2.1310
[*] Epoch 13 Completed - Accuracy: 68.89%, Epsilon: 2.85
  Epoch 14/15, Batch 200/572, Loss: 2.2243
  Epoch 14/15, Batch 400/572, Loss: 2.2878
[*] Epoch 14 Completed - Accuracy: 70.64%, Epsilon: 2.92
  Epoch 15/15, Batch 200/572, Loss: 2.3005
  Epoch 15/15, Batch 400/572, Loss: 2.2392
[*] Epoch 15 Completed - Accuracy: 69.85%, Epsilon: 3.00
[*] Saving model to dp_model.safetensors...
[*] Submitting model to the validator...

--- Validation Results ---
{'accuracy': 0.697, 'evaluation_time': 198.23, 'flag': 'HTB{svhn_pr1v4cy_sw33t_sp0t}', 'mia_advantage': 0.0232, 'passed': True}

[+] FLAG: HTB{svhn_pr1v4cy_sw33t_sp0t}
