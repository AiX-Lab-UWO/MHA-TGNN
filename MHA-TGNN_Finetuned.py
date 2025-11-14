import os
import time
import torch
from torch_geometric.data import DataLoader, Data
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GlobalAttention
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==============================
# CB-Focal (imbalance) settings
# ==============================
CB_BETA         = 0.995     # 0.99–0.999 usually
FOCAL_GAMMA     = 1.5       # 1.0–2.0
LABEL_SMOOTHING = 0.05      # 0–0.1 typically
NUM_CLASSES     = 9
EPS             = 1e-8

def make_cb_focal_criterion(class_counts,
                            num_classes,
                            beta=0.995,
                            gamma=1.5,
                            label_smoothing=0.05,
                            eps=1e-8,
                            device=None):
    """
    Class-Balanced Focal Loss with label smoothing.

    - class_counts: array-like of length C with sample counts per class
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if device is not None:
        counts = counts.to(device)

    # Class-Balanced weights (Effective Number of Samples)
    if beta is not None:
        effective_num = 1.0 - torch.pow(torch.as_tensor(beta), counts)
        weights = (1.0 - beta) / (effective_num + eps)
    else:
        weights = torch.ones_like(counts)

    # normalize weights to mean 1.0 (keeps loss scale stable)
    weights = weights * (num_classes / (weights.sum() + eps))

    def criterion(logits, targets):
        # logits: [B, C], targets: [B]
        probs = torch.softmax(logits, dim=1).clamp_min(eps)  # [B, C]

        # label smoothing: y = (1-ε)*one_hot + ε/C
        with torch.no_grad():
            y_true = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
            if label_smoothing and label_smoothing > 0:
                y_true = (1.0 - label_smoothing) * y_true + (label_smoothing / num_classes)

        # p_t = sum_c y_true_c * p_c
        p_t = (probs * y_true).sum(dim=1).clamp_min(eps)     # [B]
        focal = (1.0 - p_t).pow(gamma)                      # [B]

        # per-class alpha weighting
        alpha = weights.to(logits.device)                   # [C]
        ce_per_class = -y_true * torch.log(probs) * alpha   # [B, C]
        ce = ce_per_class.sum(dim=1)                        # [B]

        loss = (focal * ce).mean()
        return loss

    return criterion


# TransformerNet model definition
class TransformerNet(torch.nn.Module):
    def __init__(self, num_node_features, output_dim=9):
        super(TransformerNet, self).__init__()
        head_dim1 = 64
        head_dim2 = 32
        head_dim3 = 16
        head_dim4 = 8
        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8)
        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8)
        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4)
        self.conv4 = TransformerConv(head_dim3 * 4, head_dim4 * 4)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim4 * 4, 1))
        self.fc = torch.nn.Linear(head_dim4 * 4, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
        x = self.att_pool(x, data.batch)
        x = self.fc(x)
        return x


# Function to build graphs
def build_graphs(df_landmarks, edges):
    data_list = []
    for _, row in df_landmarks.iterrows():
        landmarks = row.iloc[3:].values.reshape(-1, 3)
        gaze_zone = int(row["Gaze Zone Number"])
        G = nx.Graph()

        for i, landmark in enumerate(landmarks):
            G.add_node(i, pos=landmark)

        G.add_edges_from(edges)

        x = torch.tensor([G.nodes[i]['pos'] for i in G.nodes()], dtype=torch.float)
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        y = torch.tensor([gaze_zone], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            labels.extend(data.y.cpu().numpy())
            predictions.extend(preds)
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))
    return accuracy, cm


# Main script
if __name__ == "__main__":
    data_dir = r".../S6_face_RGB"
    model_path = ".../trained_model_No_Or.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define edges
    edges = [(468, node) for node in range(478) if node != 468]  # 468 connects to all
    edges += [(473, node) for node in range(478) if node != 473]  # 473 connects to all
    edges.extend([
        (471, 159), (159, 469), (469, 145), (145, 471),
        (476, 475), (475, 474), (474, 477), (477, 476),
        (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
        (33, 246), (246, 161), (161, 160), (160, 470), (470, 158), (158, 157),
        (157, 173), (173, 155), (155, 154), (154, 153), (153, 145), (145, 144),
        (144, 163), (163, 7), (7, 33),
        (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
        (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),
        (374, 380), (380, 381), (381, 382), (382, 398)
    ])

    # Start validation from Subject 6 onward (test on 6, then 7, ..., 15)
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    total_accuracy = 0.0
    num_folds = 10  # Subjects 6 to 15

    for test_subject in range(6, 16):  # Test on subjects 6 to 15
        print(f"\nFold {test_subject - 5}/10 (Testing on Subject {test_subject})")

        # Prepare train and test datasets
        train_data, test_data = [], []
        for i in range(1, 16):  # Subjects 1 to 15
            subject_path = os.path.join(data_dir, str(i), "Landmark_RGB_Valid.csv")
            df = pd.read_csv(subject_path)

            if i == test_subject:
                test_data.extend(build_graphs(df, edges))
            else:
                train_data.extend(build_graphs(df, edges))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Load pretrained model and fine-tune
        model = TransformerNet(num_node_features=3, output_dim=NUM_CLASSES).to(device)
        pretrained_weights = torch.load(model_path, map_location=device)
        pretrained_weights.pop("fc.weight", None)
        pretrained_weights.pop("fc.bias", None)
        model.load_state_dict(pretrained_weights, strict=False)
        torch.nn.init.xavier_uniform_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)

        optimizer = Adam(model.parameters(), lr=0.001)

        # ===== Build CB-Focal criterion based on train_data label distribution =====
        counts = np.bincount(
            [int(d.y.item()) for d in train_data],
            minlength=NUM_CLASSES
        )
        criterion = make_cb_focal_criterion(
            class_counts=counts,
            num_classes=NUM_CLASSES,
            beta=CB_BETA,
            gamma=FOCAL_GAMMA,
            label_smoothing=LABEL_SMOOTHING,
            eps=EPS,
            device=device
        )

        # Training
        start_train_time = time.time()
        for epoch in range(50):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            # you can print train_loss if you want:
            # print(f"Epoch {epoch+1:03d} | loss={train_loss:.4f}")
        train_time = time.time() - start_train_time

        # Testing
        start_test_time = time.time()
        accuracy, cm = evaluate(model, test_loader, device)
        test_time = time.time() - start_test_time

        # Log results
        total_cm += cm
        total_accuracy += accuracy
        print(f"Test Subject {test_subject} Accuracy: {accuracy:.4f}")
        print(f"Training Time: {train_time:.2f} seconds, Testing Time: {test_time:.2f} seconds")
        print("Confusion Matrix:")
        print(cm)

    # Average results
    avg_accuracy = total_accuracy / num_folds
    print(f"\nAverage Accuracy Across All Folds: {avg_accuracy:.4f}")
    ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=list(range(NUM_CLASSES))).plot(cmap="viridis")
    plt.title("Average Confusion Matrix")
    plt.show()
