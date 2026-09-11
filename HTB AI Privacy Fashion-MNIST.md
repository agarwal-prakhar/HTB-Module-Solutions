Here is the complete Python script to solve the Skills Assessment. Since the challenge requires a privacy-preserving model that resists membership inference attacks (MIA), we will use **DP-SGD (Differentially Private Stochastic Gradient Descent)** again via the `opacus` library. This guarantees a low MIA advantage while easily achieving the required 70% accuracy.

HTB{pr1v4cy_d3f3nd3r_m4st3r_c0mpl3t3}

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip3 install torch torchvision opacus safetensors requests --break-system-packages
```

### Solution Script (`solve_fashion.py`)
Save the following code as `solve_fashion.py`, update the `BASE_URL` variable with your instance IP and Port, and run it:

```python
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from safetensors.torch import save_file

# --- Configuration ---
# Replace with your actual instance URL
BASE_URL = os.environ.get("BASE_URL", "http://INSTANCE_IP:PORT")

BATCH_SIZE = 128
EPOCHS = 5
LR = 0.05
MAX_GRAD_NORM = 1.0
TARGET_DELTA = 1e-5
TARGET_EPSILON = 5.0  # Low epsilon ensures MIA advantage is heavily reduced

# --- Model Architecture ---
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

print("[*] Downloading and loading Fashion-MNIST dataset...")
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

model = FashionMNISTCNN().to(device)

# Validate module for Opacus
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
print("[*] Saving model to defended_model.safetensors...")
# Access the underlying model using model._module.state_dict()
save_file(model._module.state_dict(), "defended_model.safetensors")

print(f"[*] Submitting model to {BASE_URL}/submit...")
with open("defended_model.safetensors", "rb") as f:
    try:
        response = requests.post(
            f"{BASE_URL}/submit",
            files={"defended_model": ("defended_model.safetensors", f, "application/octet-stream")}
        )
        result = response.json()
        print("\n--- Submission Results ---")
        print(result)
        if result.get("valid"):
            print(f"\n[+] FLAG: {result.get('flag')}")
    except Exception as e:
        print(f"[!] Error submitting model: {e}")
```

### How to run it:
1. Spawn your target instance in HTB and copy the IP and Port.
2. Set the `BASE_URL` variable in the script (or export it in your terminal: `export BASE_URL="http://<IP>:<PORT>"`).
3. Run the script:
```bash
python3 solve_fashion.py
```

The script will download the dataset, train the model with DP-SGD for 5 epochs, and automatically submit the `defended_model.safetensors` file to print out your success flag!

----------------------------

<img width="973" height="748" alt="image" src="https://github.com/user-attachments/assets/c49e76ef-9280-4037-8b80-445d92b518de" />

-----------------------------

HTB{pr1v4cy_d3f3nd3r_m4st3r_c0mpl3t3}
