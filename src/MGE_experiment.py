import pandas as pd
import numpy as np
import re
import networkx as nx
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import random
import argparse



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

# Function to generate kmers from a sequence
def get_kmer(sequence, k=4):
    if len(sequence) < k:
        return []  # Return an empty list if the sequence is too short
    kmers = []
    sequence = ''.join(map(str, sequence))  # Convert sequence list to string
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i + k])
    return kmers

# Function to generate De Bruijn graph edges from kmers
def get_debruijn_edges(kmers):
    edges = set()
    for k1 in kmers:
        for k2 in kmers:
            if k1 != k2 and k1[1:] == k2[:-1]:
                edges.add((k1, k2))
    return edges

# Function to create a NetworkX graph from a sequence
def sequence_to_nx_graph(sequence, k=4):
    kmers = get_kmer(sequence, k)
    if not kmers:
        return nx.DiGraph()  # Return an empty graph if no k-mers were generated
    edges = get_debruijn_edges(kmers)
    G = nx.DiGraph()
    for kmer in kmers:
        G.add_node(kmer, kmer_feature=kmer)
    G.add_edges_from(edges)
    return G

def convert_nx_to_torch_geo(G, label):
    kmer_list = list(G.nodes())
    if not kmer_list:
        return None  # Return None or handle appropriately for graphs with no nodes

    # Assuming each k-mer is a string of numbers, and we convert each character to an integer
    max_kmer_len = max(len(kmer) for kmer in kmer_list) if kmer_list else 0
    x = torch.zeros((len(kmer_list), max_kmer_len), dtype=torch.float)
    for i, kmer in enumerate(kmer_list):
        int_kmer = [int(char) for char in kmer]  # Convert each character to an integer
        x[i, :len(int_kmer)] = torch.tensor(int_kmer, dtype=torch.float)

    # Prepare edge indices
    if len(G.edges()) == 0:
        print(f"Warning: Graph with label {label} has no edges.")
        return None  # Skip graphs with no edges

    kmer_index = {kmer: idx for idx, kmer in enumerate(kmer_list)}
    edge_index = torch.tensor([[kmer_index[u], kmer_index[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

    # Prepare label
    y = torch.tensor([label], dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(G.nodes()))
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


# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Aggregates node embeddings into graph embeddings
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            
    # Training loop with detailed reporting
def train():
    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total_samples = 0

    for data in train_loader:
        if data.y.size(0) == 0:
            print("Empty batch detected")
            continue  # Skip processing if the batch is empty

        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        total_samples += data.y.size(0)
        correct += (out.argmax(dim=1) == data.y).sum().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    accuracy = correct / total_samples if total_samples > 0 else 0
    print(f"Training - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Total Batches: {num_batches}, Total Samples: {total_samples}")
    return avg_loss, accuracy
    

def test(loader, phase='Testing'):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for data in loader:
            if data.y.size(0) == 0:
                print("Empty batch detected")
                continue

            out = model(data)
            prob = F.softmax(out, dim=1)[:, 1]  # Probability of class 1
            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

            all_predictions.extend(pred.tolist())
            all_labels.extend(data.y.tolist())
            all_probabilities.extend(prob.tolist())

    accuracy = correct / total if total > 0 else 0
    roc_auc = roc_auc_score(all_labels, all_probabilities) if len(set(all_labels)) > 1 else float('nan')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"{phase} - Total Correct: {correct}, Total Samples: {total}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
    print(f"{phase} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"{phase} - Confusion Matrix:\n{cm}")

    return accuracy, roc_auc, f1, precision, recall, cm
    


def create_balanced_datasets(data_list, num_datasets=5, seed=42):
    """
    Create balanced datasets from the given list of data objects.

    Parameters:
        data_list (list): A list of data objects with labels.
        num_datasets (int): The number of balanced datasets to create.
        seed (int): The random seed for reproducibility.

    Returns:
        list: A list of balanced datasets.
    """
    # Separate the minority and dominant class samples
    minority_class = [data for data in data_list if data.y.item() == 0]
    dominant_class = [data for data in data_list if data.y.item() == 1]

    # Initialize a list to store the balanced datasets
    balanced_datasets = []

    # Seed for reproducibility
    random.seed(seed)

    # Create multiple balanced datasets
    for _ in range(num_datasets):
        # Randomly sample from the dominant class
        sampled_dominant_class = random.sample(dominant_class, len(minority_class))

        # Combine the minority class with the sampled dominant class
        balanced_dataset = minority_class + sampled_dominant_class

        # Shuffle the combined dataset
        random.shuffle(balanced_dataset)

        # Add the balanced dataset to the list
        balanced_datasets.append(balanced_dataset)

    return balanced_datasets






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Define command line arguments
    parser.add_argument('--balanced_bool', type=bool, default=False, help='Boolean flag for balance')
    parser.add_argument('--filename_fuse', type=str, default='/content/fingerprint_fuse.txt', help='Filename for fuse data')
    parser.add_argument('--filename_no_fuse', type=str, default='/content/fingerprint_no_fuse.txt', help='Filename for no fuse data')
    parser.add_argument('--k', type=int, default=4, help='An integer for k')

    # Parse arguments
    args = parser.parse_args()

    # Access command line arguments
    balanced_bool = args.balanced_bool
    filename_fuse = args.filename_fuse
    filename_no_fuse = args.filename_no_fuse
    k = args.k

    # You can now use these variables in your script
    print(f'balanced_bool: {balanced_bool}')
    print(f'filename_fuse: {filename_fuse}')
    print(f'filename_no_fuse: {filename_no_fuse}')
    print(f'k: {k}')

    #inputs
    balanced_bool = False
    filename_fuse = '/content/fingerprint_fuse.txt'
    filename_no_fuse = '/content/fingerprint_no_fuse.txt'  # Change this to the actual filename
    k = 4

    df_no_fuse = process_file(filename_no_fuse, 0)
    df_fuse = process_file(filename_fuse, 1)
    df_no_fuse = extract_numbers(df_no_fuse)
    df_fuse = extract_numbers(df_fuse)

    print(df_no_fuse.head())
    print(df_fuse.head())
    # Filter valid rows based on k-mer size
    
    df_no_fuse = filter_valid_rows(df_no_fuse, k)
    df_fuse = filter_valid_rows(df_fuse, k)

    # Extract numbers and convert each row to a graph, then to a PyTorch Geometric Data object
    data_list_no_fuse = []
    data_list_fuse = []

    for index, row in df_no_fuse.iterrows():
        sequence = row['Numbers']
        graph = sequence_to_nx_graph(sequence, k)
        data_object = convert_nx_to_torch_geo(graph, label=row['Label'])
        if data_object is not None:
            data_list_no_fuse.append(data_object)

    for index, row in df_fuse.iterrows():
        sequence = row['Numbers']
        graph = sequence_to_nx_graph(sequence, k)
        data_object = convert_nx_to_torch_geo(graph, label=row['Label'])
        if data_object is not None:
            data_list_fuse.append(data_object)

    # Combine the two lists
    all_data = data_list_no_fuse + data_list_fuse

    # Verify all graphs have nodes and edges
    filtered_data_list = [data for data in all_data if data is not None and data.x.size(0) > 0 and data.edge_index.size(1) > 0]
    print(f"Number of valid graphs after filtering: {len(filtered_data_list)}")
    
    # Set the parameters
    in_channels = k       # Each node has K features
    hidden_channels = 16  # Number of hidden units
    out_channels = 2      # Binary classification (2 classes)

    # Instantiate the model
    model = GCN(in_channels, hidden_channels, out_channels)
    early_stopping = EarlyStopping(patience=10, min_delta=0.01)


    # Define the optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


    if balanced_bool == True:
        num_datasets = 5
        balanced_datasets = create_balanced_datasets(filtered_data_list, num_datasets=5)
        # Initialize performance tracking
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        roc_aucs = []
        # Process each balanced dataset
        for i, selected_dataset in enumerate(balanced_datasets):
            print(f"\nProcessing Balanced Dataset {i+1}/{num_datasets}")

            # Perform a train-val-test split on the selected balanced dataset
            train_data, temp_data = train_test_split(
                selected_dataset,
                test_size=0.4,  # 40% goes to temp set which will be split into val and test
                random_state=42,
                stratify=[data.y.item() for data in selected_dataset]
            )

            val_data, test_data = train_test_split(
                temp_data,
                test_size=0.5,  # Split temp set equally for val and test
                random_state=42,
                stratify=[data.y.item() for data in temp_data]
            )

            # Create DataLoaders for training, validation, and testing
            batch_size = 32
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            # Train and evaluate the model with detailed reporting
            early_stopping = EarlyStopping(patience=5)

            for epoch in range(100):
                print(f"Epoch {epoch+1}/{100}")
                loss, train_acc = train()
                val_acc, _ = test(val_loader, phase='Validation')
                scheduler.step(loss)  # Adjust learning rate based on validation loss
                print(f"Epoch {epoch+1} Summary: Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n")

                # Check early stopping
                early_stopping(val_acc)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # Evaluate on the test set after training
            test_acc, roc_auc = test(test_loader, phase='Test')

            # Record the performance for each dataset
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            roc_aucs.append(roc_auc)
        
            # Calculate and print the mean performance across all datasets
        mean_train_accuracy = np.mean(train_accuracies)
        mean_val_accuracy = np.mean(val_accuracies)
        mean_test_accuracy = np.mean(test_accuracies)
        mean_roc_auc = np.nanmean(roc_aucs)  # Use nanmean to ignore NaNs from single class scenarios

        print("\nOverall Performance Across All Datasets")
        print(f"Mean Training Accuracy: {mean_train_accuracy:.4f}")
        print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f}")
        print(f"Mean Test Accuracy: {mean_test_accuracy:.4f}")
        print(f"Mean ROC AUC: {mean_roc_auc:.4f}")

    elif balanced_bool == False:
        # Split the perfectly balanced dataset into train, validation, and test sets
        train_data, temp_data = train_test_split(
            filtered_data_list,
            test_size=0.4,  # 40% goes to temp set which will be split into val and test
            random_state=42,
            stratify=[data.y.item() for data in filtered_data_list]
        )

        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,  # Split temp set equally for val and test
            random_state=42,
            stratify=[data.y.item() for data in temp_data]
        )
        # Create DataLoaders for training, validation, and testing
        batch_size = 32
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Track performances
        performance_history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_roc_auc": [],
            "val_f1_score": [],
            "val_precision": [],
            "val_recall": [],
        }
        for epoch in range(100):
            print(f"Epoch {epoch+1}/{100}")
            loss, train_acc = train()
            val_acc, val_roc_auc, val_f1, val_precision, val_recall, _ = test(val_loader, phase='Validation')
            print(f"Epoch {epoch+1} Summary: Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val ROC AUC: {val_roc_auc:.4f}\n")

            # Record performance metrics
            performance_history["epoch"].append(epoch + 1)
            performance_history["train_loss"].append(loss)
            performance_history["train_accuracy"].append(train_acc)
            performance_history["val_accuracy"].append(val_acc)
            performance_history["val_roc_auc"].append(val_roc_auc)
            performance_history["val_f1_score"].append(val_f1)
            performance_history["val_precision"].append(val_precision)
            performance_history["val_recall"].append(val_recall)

            # Check early stopping
            early_stopping(val_roc_auc)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        # Evaluate on the test set after training
        test_acc, roc_auc, f1, precision, recall, cm = test(test_loader, phase='Test')

        # Save the final test performance to the history
        performance_history["test_accuracy"] = test_acc
        performance_history["test_roc_auc"] = roc_auc
        performance_history["test_f1_score"] = f1
        performance_history["test_precision"] = precision
        performance_history["test_recall"] = recall
        performance_history["confusion_matrix"] = cm

        print("\nFinal Performance")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ROC AUC: {roc_auc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{cm}")