
import os
import torch
from torch_geometric.data import DataLoader, Data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GlobalAttention
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import networkx as nx
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# TransformerNet model definition
class TransformerNet(torch.nn.Module):
    def __init__(self, num_node_features, output_dim):
        super(TransformerNet, self).__init__()
        head_dim1 = 64
        head_dim2 = 32
        head_dim3 = 16
        self.conv1 = TransformerConv(num_node_features, head_dim1 * 8, heads=8, concat=False)
        self.conv2 = TransformerConv(head_dim1 * 8, head_dim2 * 8, heads=8, concat=False)
        self.conv3 = TransformerConv(head_dim2 * 8, head_dim3 * 4, heads=8, concat=False)
        self.att_pool = GlobalAttention(gate_nn=torch.nn.Linear(head_dim3 * 4, 1))
        self.fc = torch.nn.Linear(head_dim3 * 4, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = self.att_pool(x, data.batch)
        x = self.fc(x)
        return x


# Function to build graphs from CSV data
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


# Function to load data
def load_data(subject_path, edges):
    df = pd.read_csv(subject_path)
    return build_graphs(df, edges)


# Function to evaluate and print results
def evaluate_and_print_results(model, test_loader, device, test_type):
    accuracy, cm = evaluate(model, test_loader, device)
    print(f"\nTest Results ({test_type}):")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(9))).plot(cmap="viridis")
    plt.title(f"Confusion Matrix ({test_type})")
    plt.show()


# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu().numpy()
            labels.extend(data.y.cpu().numpy())
            predictions.extend(preds)
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=list(range(9)))
    return accuracy, cm


# Main script
if __name__ == "__main__":
    data_dir = r"...\DMD\S6_face_RGB"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Edges for the graph
    edges = [(468, node) for node in range(478) if node != 468]
    edges += [(473, node) for node in range(478) if node != 473]
    edges.extend([(471, 159), (159, 469), (469, 145), (145, 471), (476, 475), (475, 474), (474, 477), (477, 476),
                  (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368), (33, 246), (146, 161), (161, 160),
                  (160, 150), (150, 158), (158, 157), (157, 173), (173, 155), (155, 154), (154, 153), (153, 145),
                  (145, 144), (144, 163), (163, 7), (7, 33), (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
                  (388, 263), (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382),
                  (382, 398)])

    # Cross-validation setup
    num_subjects = 15
    for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=num_subjects).split(range(1, num_subjects + 1))):
        print(f"Fold {fold + 1}/{num_subjects}")

        # Prepare train and test datasets
        train_data_rgb, train_data_ir = [], []
        test_data_rgb, test_data_ir = [], []

        for i in train_idx:
            subject_path_rgb = os.path.join(data_dir, str(i + 1), "Landmark_RGB_Valid.csv")
            subject_path_ir = os.path.join(data_dir, str(i + 1), "Landmark_IR_Valid.csv")
            train_data_rgb.extend(load_data(subject_path_rgb, edges))
            train_data_ir.extend(load_data(subject_path_ir, edges))

        for i in test_idx:
            subject_path_rgb = os.path.join(data_dir, str(i + 1), "Landmark_RGB_Valid.csv")
            subject_path_ir = os.path.join(data_dir, str(i + 1), "Landmark_IR_Valid.csv")
            test_data_rgb.extend(load_data(subject_path_rgb, edges))
            test_data_ir.extend(load_data(subject_path_ir, edges))

        # Data loaders
        train_loader_rgb = DataLoader(train_data_rgb, batch_size=32, shuffle=True)
        train_loader_ir = DataLoader(train_data_ir, batch_size=32, shuffle=True)
        test_loader_rgb = DataLoader(test_data_rgb, batch_size=32, shuffle=False)
        test_loader_ir = DataLoader(test_data_ir, batch_size=32, shuffle=False)

        # Scenario 1: Train on RGB, Test on IR
        print("\nScenario 1: Train on RGB, Test on IR")
        model = TransformerNet(num_node_features=3, output_dim=9).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(50):
            train_loss = train(model, train_loader_rgb, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")
            scheduler.step()

        # Evaluation
        evaluate_and_print_results(model, test_loader_ir, device, "Train on RGB, Test on IR")

        # Scenario 2: Train on IR, Test on RGB
        print("\nScenario 2: Train on IR, Test on RGB")
        model = TransformerNet(num_node_features=3, output_dim=9).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        # Training loop
        for epoch in range(50):
            train_loss = train(model, train_loader_ir, optimizer, criterion, device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")
            scheduler.step()

        # Evaluation
        evaluate_and_print_results(model, test_loader_rgb, device, "Train on IR, Test on RGB")
