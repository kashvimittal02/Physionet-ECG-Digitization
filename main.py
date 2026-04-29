import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_DIR = "train"
TEST_DIR = "test"
IMG_SIZE = 256
SIGNAL_LEN = 1024
BATCH_SIZE = 8
EPOCHS = 25
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# DATASET
# -----------------------------
class ECGDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for folder in os.listdir(root_dir):
            path = os.path.join(root_dir, folder)
            if not os.path.isdir(path):
                continue

            csv_path = os.path.join(path, f"{folder}.csv")
            if not os.path.exists(csv_path):
                continue

            gt = pd.read_csv(csv_path)
            signal = gt.select_dtypes(include=['number']).iloc[:, 0].values.astype(np.float32)

            signal = signal[~np.isnan(signal)]
            if len(signal) < 10:
                continue

            signal = savgol_filter(signal, 11, 2)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            signal = np.interp(
                np.linspace(0, len(signal) - 1, SIGNAL_LEN),
                np.arange(len(signal)),
                signal
            )

            for f in os.listdir(path):
                if f.endswith(".png"):
                    self.samples.append((os.path.join(path, f), signal))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, signal = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        h = img.shape[0]
        img = img[3*h//4:h, :]

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
        img = img + edges / 255.0

        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0)

        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(signal, dtype=torch.float32)

# -----------------------------
# MODEL
# -----------------------------
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.flatten = nn.Flatten(2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(128 * 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, SIGNAL_LEN)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# -----------------------------
# UTILS
# -----------------------------
def evaluate_model(model, dataset):
    model.eval()
    total_loss = 0
    count = min(50, len(dataset))

    with torch.no_grad():
        for i in range(count):
            img, gt = dataset[i]
            img = img.unsqueeze(0).to(DEVICE)
            gt = gt.to(DEVICE)

            pred = model(img).squeeze()
            loss = torch.mean((pred - gt) ** 2)
            total_loss += loss.item()

    return total_loss / count


def evaluate_full_metrics(model, dataset):
    model.eval()
    preds_all = []
    gts_all = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img, gt = dataset[i]
            img = img.unsqueeze(0).to(DEVICE)

            pred = model(img).cpu().numpy().flatten()
            gt = gt.numpy().flatten()

            preds_all.extend(pred)
            gts_all.extend(gt)

    preds_all = np.array(preds_all)
    gts_all = np.array(gts_all)

    mse = mean_squared_error(gts_all, preds_all)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(gts_all, preds_all)
    mape = np.mean(np.abs((gts_all - preds_all) / (gts_all + 1e-8))) * 100
    r2 = r2_score(gts_all, preds_all)

    print("\n===== FULL METRICS =====")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R2   : {r2:.4f}")

    return mse


def baseline_model(dataset):
    preds_all = []
    gts_all = []

    for i in range(len(dataset)):
        _, gt = dataset[i]
        gt = gt.numpy()
        pred = np.ones_like(gt) * np.mean(gt)

        preds_all.extend(pred)
        gts_all.extend(gt)

    preds_all = np.array(preds_all)
    gts_all = np.array(gts_all)

    mse = np.mean((preds_all - gts_all)**2)
    return mse


def plot_prediction(model, dataset):
    model.eval()
    img, gt = dataset[0]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()

    plt.figure(figsize=(10,4))
    plt.plot(gt.numpy(), label="GT")
    plt.plot(pred, label="Pred", alpha=0.7)
    plt.legend()
    plt.title("Prediction vs Ground Truth")
    plt.show()

# -----------------------------
# TRAINING
# -----------------------------
dataset = ECGDataset(TRAIN_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ECGModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

loss_history = []

print(f"Training on {len(dataset)} samples")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, signals in loader:
        imgs = imgs.to(DEVICE)
        signals = signals.to(DEVICE)

        preds = model(imgs)
        preds = torch.clamp(preds, -5, 5)

        mse = (preds - signals) ** 2
        peak_weight = 1 + torch.abs(signals)
        loss = (mse * peak_weight).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    epoch_loss = total_loss/len(loader)
    loss_history.append(epoch_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "ecg_model.pth")
print("Model saved")

# -----------------------------
# LOSS CURVE
# -----------------------------
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# -----------------------------
# EVALUATION
# -----------------------------
val_loss = evaluate_model(model, dataset)
print(f"Validation MSE: {val_loss:.4f}")

plot_prediction(model, dataset)

model_mse = evaluate_full_metrics(model, dataset)
baseline_mse = baseline_model(dataset)

print("\n===== BASELINE COMPARISON =====")
print(f"Baseline MSE: {baseline_mse:.4f}")
print(f"Model MSE   : {model_mse:.4f}")

# -----------------------------
# TEST / SUBMISSION
# -----------------------------
test_meta = pd.read_csv("test.csv")
rows = []

for img_id in test_meta["id"].unique():
    img_path = os.path.join(TEST_DIR, f"{img_id}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h = img.shape[0]
    img = img[3*h//4:h, :]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = (img - 0.5) / 0.5
    img = np.expand_dims(img, axis=0)

    inp = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(inp).cpu().numpy().flatten()

    subset = test_meta[test_meta["id"] == img_id]

    for _, row in subset.iterrows():
        length = row["number_of_rows"]
        sig = np.interp(
            np.linspace(0, len(pred)-1, length),
            np.arange(len(pred)),
            pred
        )

        sig = sig * (1 + 0.3 * np.abs(sig))
        sig = savgol_filter(sig, 11, 2)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

        sig_str = " ".join(map(str, sig))
        rows.append([img_id, row["lead"], sig_str])

submission = pd.DataFrame(rows, columns=["id", "lead", "signal"])
submission.to_csv("submission.csv", index=False)

print("submission.csv generated")