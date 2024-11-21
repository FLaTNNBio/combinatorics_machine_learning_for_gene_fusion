import pandas as pd
import numpy as np
import re
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HypergraphConv, global_mean_pool
import gc  # Import garbage collection module
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import argparse
import os



# Function to read and process the file
def process_file(filename, label):
    with open(filename, 'r') as file:
        data = file.read()
    entries = data.split("', '")
    entries = [entry.strip(" '") for entry in entries]
    df = pd.DataFrame(entries, columns=["Entry"])
    df['Label'] = label
    return df

# Function to extract numeric data from entries
def extract_numbers(df):
    def extract_numeric_part(entry):
        match = re.search(r'\|([\d\s|]+)', entry)
        if match:
            return ' '.join(match.group(1).strip().split('|')).strip()
        return ""
    numeric_data = df['Entry'].apply(extract_numeric_part)
    numeric_data = numeric_data.apply(lambda x: list(map(int, x.split())) if x else [])
    return pd.DataFrame({'Numbers': numeric_data, 'Label': df['Label']})

# Function to create a De Bruijn hypergraph from fingerprints
def create_de_bruijn_hypergraph(fingerprint, k=4):
    G = nx.Graph()
    nodes = []
    k_fingers = []

    # Generate k-fingers from the fingerprint
    for i in range(len(fingerprint) - k + 1):
        k_finger = tuple(fingerprint[i:i + k])
        if k_finger not in nodes:
            nodes.append(k_finger)
        k_fingers.append(k_finger)

    # Add nodes to the graph
    node_indices = {k_finger: idx for idx, k_finger in enumerate(nodes)}
    for k_finger in nodes:
        G.add_node(node_indices[k_finger], label=k_finger)

    # Form hyperedges based on overlaps
    for i in range(len(k_fingers) - 1):
        if k_fingers[i][1:] == k_fingers[i + 1][:-1]:
            G.add_edge(node_indices[k_fingers[i]], node_indices[k_fingers[i + 1]])

    return G, node_indices

# Convert k-mer tuples to numerical features
def kmer_to_features(kmer, max_kmer_length):
    # Convert k-mer into a fixed-length numerical vector
    return torch.tensor([float(x) for x in kmer], dtype=torch.float)

# Convert NetworkX graph to PyG Data
def convert_hypergraph_to_data(G, nodes, label):
    # Create edge index
    edge_index_list = list(G.edges)

    if len(edge_index_list) == 0:
        print(f"Warning: Hypergraph with label {label} has no edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge index
    else:
        edge_index = torch.tensor(edge_index_list).t().contiguous()

    # Create node features from k-mers
    x = torch.stack([kmer_to_features(G.nodes[i]['label'], len(G.nodes[i]['label'])) for i in range(len(nodes))])

    y = torch.tensor([label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(G.nodes))
    return data

# Pre-process checks to filter out invalid sequences
def filter_valid_rows(df, k):
    valid_rows = []
    for index, row in df.iterrows():
        sequence = row['Numbers']
        if len(sequence) >= k:
            valid_rows.append(row)
        else:
            print(f"Skipping row {index} due to insufficient sequence length.", sequence)
    return pd.DataFrame(valid_rows)

# Define the HGCN Model
class HGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HGCN, self).__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Pool the node embeddings to obtain graph embeddings
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(loader):
    model.eval()
    correct = 0
    predictions = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            prob = F.softmax(out, dim=1)[:, 1]  # Probability of the positive class
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
            probabilities.extend(prob.cpu().numpy())
            correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), predictions, labels, probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Define command line arguments
    parser.add_argument('--filename_fuse', type=str, help='Filename for fuse data')
    parser.add_argument('--filename_no_fuse', type=str, help='Filename for no fuse data')
    parser.add_argument('--k', type=int, default=4, help='An integer for k')

    # Parse arguments
    args = parser.parse_args()

    # Access command line arguments
    filename_fuse = args.filename_fuse
    filename_no_fuse = args.filename_no_fuse
    k = args.k

    # You can now use these variables in your script
    print(f'filename_fuse: {filename_fuse}')
    print(f'filename_no_fuse: {filename_no_fuse}')
    print(f'k: {k}')

    df_no_fuse = process_file(filename_no_fuse, 0)
    df_fuse = process_file(filename_fuse, 1)
    df_no_fuse = extract_numbers(df_no_fuse)
    df_fuse = extract_numbers(df_fuse)

    print(df_no_fuse.head())
    print(df_fuse.head())

    # Filter valid rows based on k-mer size

    df_no_fuse = filter_valid_rows(df_no_fuse, k)
    df_fuse = filter_valid_rows(df_fuse, k)

    # Convert each row to a hypergraph, then to a PyTorch Geometric Data object
    data_list_no_fuse = []
    data_list_fuse = []

    for index, row in df_no_fuse.iterrows():
        fingerprint = row['Numbers']  # Use the sequence directly as fingerprints
        graph, nodes = create_de_bruijn_hypergraph(fingerprint, k)
        data_object = convert_hypergraph_to_data(graph, nodes, label=row['Label'])
        if data_object is not None:
            data_list_no_fuse.append(data_object)

    for index, row in df_fuse.iterrows():
        fingerprint = row['Numbers']  # Use the sequence directly as fingerprints
        graph, nodes = create_de_bruijn_hypergraph(fingerprint, k)
        data_object = convert_hypergraph_to_data(graph, nodes, label=row['Label'])
        if data_object is not None:
            data_list_fuse.append(data_object)

    # Combine the two lists
    all_data = data_list_no_fuse + data_list_fuse

    # Verify all graphs have nodes and edges
    filtered_data_list = [data for data in all_data if data is not None and data.x.size(0) > 0 and (data.edge_index.size(1) > 0 or data.num_nodes == 1)]
    print(f"Number of valid hypergraphs after filtering: {len(filtered_data_list)}")

    # Print a summary of the transformed graphs
    for i, data in enumerate(filtered_data_list[:5]):  # Print only the first 5 for brevity
        print(f"Graph {i+1}:")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.edge_index.size(1)}")
        print(f"  Label: {data.y.item()}")
        print("  Node features (first 5 nodes):")
        print(data.x[:5])
        print("  Edge index (first 5 edges):")
        print(data.edge_index[:, :5])
        print("\n")

    # Split the data into training, validation, and testing sets
    train_data, temp_data = train_test_split(
        filtered_data_list, test_size=0.4, random_state=42, stratify=[data.y.item() for data in filtered_data_list]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=[data.y.item() for data in temp_data]
    )
    # Create DataLoaders for training, validation, and testing
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # Initialize the model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set in_channels to k to match k-mer size
    model = HGCN(in_channels=k, hidden_channels=16, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    # Main training loop with early stopping
    num_epochs = 100
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)
    for epoch in range(1, num_epochs + 1):
        train_loss = train()
        train_acc, _, _, _ = evaluate(train_loader)
        val_acc, _, _, _ = evaluate(val_loader)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Check early stopping condition
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    # Evaluate on the test set
    test_acc, predictions, labels, probabilities = evaluate(test_loader)

    # Calculate performance metrics
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    roc_auc = roc_auc_score(labels, probabilities)
    conf_matrix = confusion_matrix(labels, predictions)
    # Save and print results
    performance_metrics = {
        'Accuracy': test_acc,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC AUC': roc_auc,
        'Confusion Matrix': conf_matrix
    }

    # Print the results
    print("\nFinal Test Performance Metrics:")
    for metric, value in performance_metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")

    save_path = './saved_models/'  # Change this to your desired folder
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'model_MLL.pth'))
    print(f"Model saved in '{save_path}' as 'model_MLL.pth'")

