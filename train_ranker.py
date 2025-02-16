import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn.functional as F

dataset_path = "/Users/saket/Documents/CMU/CMR/ConeClassifier/dataset"

# 1. Load the dataset (boundaries and cone maps)
def load_yaml_data(path):
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)
    
def pad_sequence(sequence, max_len):
    """
    Pad sequences to the maximum length in the batch.
    """
    # Padding the sequence to the required size with zeros (or another padding value)
    padded_sequence = F.pad(sequence, (0, 0, 0, max_len - sequence.size(0)), value=0)
    return padded_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch to the same length.
    """
    left_tensors = [item[0] for item in batch]
    right_tensors = [item[1] for item in batch]
    
    # Find the maximum length of the sequences
    max_left_len = max([t.size(0) for t in left_tensors])
    max_right_len = max([t.size(0) for t in right_tensors])
    
    # Pad all sequences to the same length
    left_tensors = [pad_sequence(t, max_left_len) for t in left_tensors]
    right_tensors = [pad_sequence(t, max_right_len) for t in right_tensors]
    
    # Stack the tensors into a single batch
    left_tensor_batch = torch.stack(left_tensors, dim=0)
    right_tensor_batch = torch.stack(right_tensors, dim=0)
    
    return left_tensor_batch, right_tensor_batch

# Load all boundaries and cone maps
boundary_paths = [f"{dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)]
cone_map_paths = [f"{dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)]

boundaries = [load_yaml_data(path) for path in boundary_paths]
cone_maps = [load_yaml_data(path) for path in cone_map_paths]

# 2. Preprocess the data
def generate_perceptual_field_data(boundaries, cone_maps, perceptual_range=30, noise_rate=0.1):
    perceptual_field_data = []
    for boundary, cone_map in zip(boundaries, cone_maps):
        left_boundary = boundary['left']
        right_boundary = boundary['right']
        
        # Filter out points outside perceptual range
        filtered_left = filter_points_within_range(left_boundary, cone_map, perceptual_range)
        filtered_right = filter_points_within_range(right_boundary, cone_map, perceptual_range)
        
        # Add noise for false positives
        noisy_left = add_noise(filtered_left, noise_rate)
        noisy_right = add_noise(filtered_right, noise_rate)
        
        perceptual_field_data.append((noisy_left, noisy_right))
    
    return perceptual_field_data

def filter_points_within_range(boundary, cone_map, perceptual_range):
    filtered = []
    for point in boundary:
        x, y = cone_map.get(point)
        if x**2 + y**2 <= perceptual_range**2:  # Check if the point is within the perceptual range
            filtered.append([x, y])
    return filtered

def add_noise(points, noise_rate):
    noisy_points = []
    for point in points:
        if random.random() < noise_rate:
            # Simulate a random false positive by adding noise to the point
            noise = np.random.normal(0, 1, size=2)
            noisy_points.append([point[0] + noise[0], point[1] + noise[1]])
        else:
            noisy_points.append(point)
    return noisy_points

# 3. Create custom dataset class
class LaneDetectionDataset(Dataset):
    def __init__(self, perceptual_field_data):
        self.data = perceptual_field_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_left, noisy_right = self.data[idx]
        left_tensor = torch.tensor(noisy_left, dtype=torch.float32)
        right_tensor = torch.tensor(noisy_right, dtype=torch.float32)
        return left_tensor, right_tensor

# 4. Define the model architecture with 1001 parameters and layers
class ConeClassifier(nn.Module):
    def __init__(self):
        super(ConeClassifier, self).__init__()
        # Simpler architecture focused on spatial relationships
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 2, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Ensure input is 3D: [batch_size, channels, sequence_length]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)  # [batch, 2, points]
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def compute_iou(pred, gt):
    """Compute IoU between predicted and ground truth points"""
    # Convert to sets of points for IoU calculation
    pred_set = set(map(tuple, pred.reshape(-1, 2)))
    gt_set = set(map(tuple, gt.reshape(-1, 2)))
    
    # Calculate intersection and union
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    
    return intersection / max(union, 1)  # Avoid division by zero

def train_model(dataset, model, epochs=100, batch_size=32, learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for left_tensor, right_tensor in dataloader:
            batch_size = left_tensor.size(0)
            optimizer.zero_grad()
            
            # Create labels: 0 for left cones, 1 for right cones
            labels_left = torch.zeros(batch_size, dtype=torch.long)
            labels_right = torch.ones(batch_size, dtype=torch.long)
            
            # Forward pass
            left_pred = model(left_tensor)
            right_pred = model(right_tensor)
            
            # Calculate loss
            loss_left = criterion(left_pred, labels_left)
            loss_right = criterion(right_pred, labels_right)
            loss = loss_left + loss_right
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_left = torch.max(left_pred.data, 1)
            _, predicted_right = torch.max(right_pred.data, 1)
            total += batch_size * 2
            correct += (predicted_left == labels_left).sum().item()
            correct += (predicted_right == labels_right).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

def evaluate_model(model, dataset):
    model.eval()
    correct = 0
    total = 0
    iou_scores = []
    
    with torch.no_grad():
        for left_tensor, right_tensor in dataset:
            if not isinstance(left_tensor, torch.Tensor):
                left_tensor = torch.tensor(left_tensor, dtype=torch.float32)
            if not isinstance(right_tensor, torch.Tensor):
                right_tensor = torch.tensor(right_tensor, dtype=torch.float32)
            
            # Get predictions
            left_pred = model(left_tensor)
            right_pred = model(right_tensor)
            
            # Calculate accuracy
            _, predicted_left = torch.max(left_pred.data, 1)
            _, predicted_right = torch.max(right_pred.data, 1)
            total += 2  # Two predictions per sample
            correct += (predicted_left == 0).sum().item()  # Should predict left (0)
            correct += (predicted_right == 1).sum().item()  # Should predict right (1)
            
            # Calculate IoU
            iou_left = compute_iou(left_tensor.numpy(), left_tensor.numpy())
            iou_right = compute_iou(right_tensor.numpy(), right_tensor.numpy())
            iou_scores.extend([iou_left, iou_right])
    
    accuracy = 100 * correct / total
    avg_iou = np.mean(iou_scores)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average IoU: {avg_iou:.4f}")

# 7. Main function to orchestrate training
def main():
    perceptual_field_data = generate_perceptual_field_data(boundaries, cone_maps)
    dataset = LaneDetectionDataset(perceptual_field_data)
    
    model = ConeClassifier()
    train_model(dataset, model)
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()