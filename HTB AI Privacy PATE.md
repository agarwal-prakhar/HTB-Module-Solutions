Here is the complete Python script to solve the PATE Challenge. It implements the PATE (Private Aggregation of Teacher Ensembles) framework, trains the student model on EMNIST Letters, and automatically submits it to the validator to retrieve your flag.

HTB{3mn1st_l3tt3rs_p4t3_m4st3ry}

### Prerequisites
Make sure you have the required libraries installed:
```bash
pip3 install torch torchvision scikit-learn safetensors requests --break-system-packages
```

### Solution Script (`solve_pate.py`)
Save the following code as `solve_pate.py`, update the `BASE_URL` variable with your instance IP and Port, and run it:

```python
import os
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
# Replace with your actual instance URL
BASE_URL = os.environ.get("BASE_URL", "http://INSTANCE_IP:PORT")

BATCH_SIZE = 128
EPOCHS = 10
N_TEACHERS = 10
NOISE_SCALE = 0.5  # Laplace noise scale for PATE aggregation

# --- Model Architecture ---
class MLP(nn.Module):
    """Multi-Layer Perceptron for EMNIST Letters classification."""
    def __init__(self, input_size=784, hidden_layers=None, num_classes=26, dropout=0.2):
        super(MLP, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128]

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropouts.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.output = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        return self.output(x)

# --- Data Preparation ---
print("[*] Downloading and loading EMNIST dataset...")
train_dataset = datasets.EMNIST("data", split='letters', train=True, download=True)
test_dataset = datasets.EMNIST("data", split='letters', train=False, download=True)

X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0

# Labels are 1-26 for A-Z, convert to 0-25
y_train = train_dataset.targets.numpy() - 1
y_test = test_dataset.targets.numpy() - 1

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# --- PATE Framework ---
# 1. Split training data into Teacher data and Public (unlabeled) data
n_total = len(X_train_t)
n_public = n_total // 5  # 20% for student training
n_teacher_data = n_total - n_public

perm = torch.randperm(n_total)
teacher_indices = perm[:n_teacher_data]
public_indices = perm[n_teacher_data:]

X_teacher = X_train_t[teacher_indices]
y_teacher = y_train_t[teacher_indices]
X_public = X_train_t[public_indices]
y_public = y_train_t[public_indices] # Only used to verify teacher agreement

# 2. Train Teachers
print(f"[*] Training {N_TEACHERS} teacher models...")
teacher_data_size = len(X_teacher) // N_TEACHERS
teachers = []

for i in range(N_TEACHERS):
    start = i * teacher_data_size
    end = (i + 1) * teacher_data_size if i != N_TEACHERS - 1 else len(X_teacher)
    
    X_subset = X_teacher[start:end]
    y_subset = y_teacher[start:end]
    
    train_ds = TensorDataset(X_subset, y_subset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Using standard MLP for teachers
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    teachers.append(model)
    print(f"  Teacher {i+1}/{N_TEACHERS} trained.")

# 3. Aggregate Teacher Predictions with DP Noise (PATE)
print("[*] Aggregating teacher predictions with DP noise...")
predictions = []
with torch.no_grad():
    for model in teachers:
        model.eval()
        preds = []
        for j in range(0, len(X_public), BATCH_SIZE):
            X_batch = X_public[j:j+BATCH_SIZE]
            outputs = model(X_batch)
            preds.append(torch.argmax(outputs, dim=1))
        predictions.append(torch.cat(preds))

# Stack predictions: shape (N_TEACHERS, n_public)
predictions = torch.stack(predictions)

n_classes = 26
noisy_labels = []
for i in range(len(X_public)):
    votes = torch.zeros(n_classes)
    for t in range(N_TEACHERS):
        votes[predictions[t, i]] += 1
    # Add Laplace noise to the vote counts to ensure DP
    noise = torch.tensor(np.random.laplace(0, NOISE_SCALE, n_classes), dtype=torch.float32)
    noisy_votes = votes + noise
    noisy_label = torch.argmax(noisy_votes).item()
    noisy_labels.append(noisy_label)

noisy_labels = torch.tensor(noisy_labels, dtype=torch.long)

# Check accuracy of noisy labels
agreement = (noisy_labels == y_public).float().mean()
print(f"[*] Noisy labels agreement with true labels: {agreement*100:.2f}%")

# 4. Train Student Model
print("[*] Training student model...")
# We use dropout=0.4 and weight_decay to regularize the student model heavily. 
# This prevents overfitting and minimizes MIA advantage.
student_model = MLP(dropout=0.4)
optimizer = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)

student_ds = TensorDataset(X_public, noisy_labels)
student_loader = DataLoader(student_ds, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    student_model.train()
    running_loss = 0.0
    for X_batch, y_batch in student_loader:
        optimizer.zero_grad()
        outputs = student_model(X_batch)
        loss = F.cross_entropy(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Evaluate on test set
    student_model.eval()
    with torch.no_grad():
        outputs = student_model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean().item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(student_loader):.4f} - Test Acc: {accuracy*100:.2f}%")

# 5. Save and Submit
print("[*] Saving student model to pate_student.safetensors...")
save_file(student_model.state_dict(), "pate_student.safetensors")

print(f"[*] Submitting model to {BASE_URL}/validate...")
with open("pate_student.safetensors", "rb") as f:
    try:
        response = requests.post(
            f"{BASE_URL}/validate",
            files={"model": ("pate_student.safetensors", f, "application/octet-stream")}
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
1. Spawn your target instance in HTB and copy the IP and Port.
2. Set the `BASE_URL` variable in the script (or export it in your terminal: `export BASE_URL="http://<IP>:<PORT>"`).
3. Run the script:
```bash
python3 solve_pate.py
```

The script will download the dataset, train the 10 teacher models, aggregate their predictions with Laplace noise, train the student model, and automatically submit the `pate_student.safetensors` file to print out your success flag!


--------------------
Perfect! I can see exactly what is taking up all your space. The biggest culprits are the partially downloaded EMNIST dataset and the `pip` cache (which is holding onto gigabytes of downloaded installation files that you no longer need).

Run these commands to safely free up several gigabytes of space:

### Step 1: Delete the partial EMNIST dataset
```bash
rm -rf /home/htb-ac-2259706/data/
```

### Step 2: Clear your pip cache
This will safely delete all the cached installation files (like the 500MB+ PyTorch and NVIDIA files) that are just sitting in your cache doing nothing.
```bash
rm -rf /home/htb-ac-2259706/.cache/pip/
```

### Step 3: (Optional) Clear your apt cache
If you want to free up a little more space, you can clear the package manager cache:
```bash
sudo apt-get clean
```

### Step 4: Verify your space
Run `df -h` again. You should now have several gigabytes available under the `Avail` column for `/dev/vda1`.

```bash
df -h
```

### Step 5: Rerun the script
Once you have at least 2GB free, you can run your script again:
```bash
python3 solve_pate.py
```

---------------------------

<img width="969" height="765" alt="image" src="https://github.com/user-attachments/assets/6f494a99-ee58-46fa-94a7-7d0ba99cde6a" />


--------------------
└──╼ [★]$ python3 solve_pate.py 
[*] Downloading and loading EMNIST dataset...
100%|████████████████████████████████████████| 562M/562M [00:10<00:00, 52.7MB/s]
[*] Training 10 teacher models...
  Teacher 1/10 trained.
  Teacher 2/10 trained.
  Teacher 3/10 trained.
  Teacher 4/10 trained.
  Teacher 5/10 trained.
  Teacher 6/10 trained.
  Teacher 7/10 trained.
  Teacher 8/10 trained.
  Teacher 9/10 trained.
  Teacher 10/10 trained.
[*] Aggregating teacher predictions with DP noise...
[*] Noisy labels agreement with true labels: 85.44%
[*] Training student model...
Epoch 1/10 - Loss: 1.4001 - Test Acc: 72.75%
Epoch 2/10 - Loss: 0.7720 - Test Acc: 76.91%
Epoch 3/10 - Loss: 0.6435 - Test Acc: 79.16%
Epoch 4/10 - Loss: 0.5776 - Test Acc: 79.93%
Epoch 5/10 - Loss: 0.5293 - Test Acc: 80.65%
Epoch 6/10 - Loss: 0.4884 - Test Acc: 81.23%
Epoch 7/10 - Loss: 0.4692 - Test Acc: 80.98%
Epoch 8/10 - Loss: 0.4525 - Test Acc: 81.95%
Epoch 9/10 - Loss: 0.4331 - Test Acc: 82.03%
Epoch 10/10 - Loss: 0.4246 - Test Acc: 81.93%
[*] Saving student model to pate_student.safetensors...
[*] Submitting model to http://x.x.x.x:y/validate...

--- Validation Results ---
{'accuracy': 0.8193, 'evaluation_time': 11.61, 'flag': 'HTB{3mn1st_l3tt3rs_p4t3_m4st3ry}', 'mia_advantage': 0.0142, 'passed': True}

[+] FLAG: HTB{3mn1st_l3tt3rs_p4t3_m4st3ry}
