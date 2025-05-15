
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
    cm = confusion_matrix(labels, predictions, labels=list(range(9)))
    return accuracy, cm


# Main script
if __name__ == "__main__":
    data_dir = r".../S6_face_RGB"
    model_path = ".../trained_model_No_Or.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define edges
    edges = [(468, node) for node in range(478) if node != 468]  # 468 connects to all
    edges += [(473, node) for node in range(478) if node != 473]  # 473 connects to all
    edges.extend([(471, 159), (159, 469), (469, 145), (145, 471),
                  (476, 475), (475, 474), (474, 477), (477, 476),
                  (1, 33), (1, 173), (1, 162), (1, 263), (1, 398), (1, 368),
                  (33, 246), (246, 161), (161, 160), (160, 470), (470, 158), (158, 157),
                  (157, 173), (173, 155), (155, 154), (154, 153), (153, 145), (145, 144),
                  (144, 163), (163, 7), (7, 33),
                  (398, 384), (384, 385), (385, 386), (386, 387), (387, 388),
                  (388, 263), (263, 249), (249, 390), (390, 373), (373, 374),
                  (374, 380), (380, 381), (381, 382), (382, 398)])

    # Start validation from Subject 6 onward (test on 6, then 7, ..., 15)
    total_cm = np.zeros((9, 9), dtype=int)
    total_accuracy = 0
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
        model = TransformerNet(num_node_features=3, output_dim=9).to(device)
        pretrained_weights = torch.load(model_path, map_location=device)
        pretrained_weights.pop("fc.weight", None)
        pretrained_weights.pop("fc.bias", None)
        model.load_state_dict(pretrained_weights, strict=False)
        torch.nn.init.xavier_uniform_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)

        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Training
        start_train_time = time.time()
        for epoch in range(50):
            train_loss = train(model, train_loader, optimizer, criterion, device)
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
    ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=list(range(9))).plot(cmap="viridis")
    plt.title("Average Confusion Matrix")
    plt.show()
