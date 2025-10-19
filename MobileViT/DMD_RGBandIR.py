import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import warnings
import matplotlib.pyplot as plt
import timm  # for MobileViT
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table


warnings.filterwarnings("ignore", category=UserWarning)

# Dataset definition
class GazeDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        frame_index = self.labels_df.iloc[idx]['Frame Index']
        gaze_label = int(self.labels_df.iloc[idx]['Gaze Zone'])
        image_path = os.path.join(self.image_dir, f"frame_{int(frame_index):04d}.jpg")
        image = plt.imread(image_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if self.transform:
            image = self.transform(image)
        return image, gaze_label

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            labels.extend(labels_batch.cpu().numpy())
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=list(range(9)))
    return accuracy, cm

# Main script
if __name__ == "__main__":
    base_dir = r"...\\DMD\\S6_face_RGB"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    subjects = list(range(1, 16))
    kfold = KFold(n_splits=15, shuffle=True, random_state=42)

    avg_cm = np.zeros((9, 9), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(subjects), start=1):
        print(f"\nFold {fold}/15")

        train_datasets = []
        test_datasets = []

        for idx in train_idx:
            subject_path = os.path.join(base_dir, str(subjects[idx]))
            train_datasets.append(GazeDataset(
                os.path.join(subject_path, "Frames_IR_Valid"),
                os.path.join(subject_path, f"Valid_gaze_label_{subjects[idx]}.csv"),
                transform))

        for idx in test_idx:
            subject_path = os.path.join(base_dir, str(subjects[idx]))
            test_datasets.append(GazeDataset(
                os.path.join(subject_path, "Frames_IR_Valid"),
                os.path.join(subject_path, f"Valid_gaze_label_{subjects[idx]}.csv"),
                transform))

        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=32, shuffle=True)
        test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=32, shuffle=False)

        model = timm.create_model('mobilevit_s', pretrained=True, num_classes=9)
        model.to(device)

        model = timm.create_model('mobilevit_s', pretrained=True, num_classes=9)
        model.to(device)

        # Dummy input for FLOP and memory analysis
        dummy_input = torch.randn(1, 3, 256, 256).to(device)

        # Parameter summary
        print(summary(model, input_size=(1, 3, 256, 256)))

        # FLOPs analysis
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
        print(parameter_count_table(model))

        # Inference memory usage (approximation)
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy_input)
            mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            if mem:
                print(f"Inference Memory Usage (approx.): {mem:.2f} MB")
            else:
                print("Inference Memory Usage: CUDA not available")

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        start_train = time.time()
        for epoch in range(50):
            loss = train(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        end_train = time.time()

        print(f"Training time for fold {fold}: {end_train - start_train:.2f} seconds")

        start_test = time.time()
        acc, cm = evaluate(model, test_loader, device)
        end_test = time.time()

        print(f"Test time for fold {fold}: {end_test - start_test:.2f} seconds")
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(f"Fold {fold} Confusion Matrix:\n{cm}")

        avg_cm += cm.astype(np.float32)

    avg_cm /= 15
    print("\nAverage Confusion Matrix Over 15 Folds of IR on IR:")
    print(avg_cm.astype(int))
