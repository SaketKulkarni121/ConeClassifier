import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import random
import torch.nn.functional as F

dataset_path = "/Users/ishaan/Documents/Projects/ConeClassifier/dataset"

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

def add_noise(points, noise_rate, perceptual_range=30, false_positive_rate=0.1):
    noisy_points = []
    for point in points:
        if random.random() < noise_rate:
            # Simulate a random false positive by adding noise to the point
            noise = np.random.normal(0, 1, size=2)
            noisy_points.append([point[0] + noise[0], point[1] + noise[1]])
        else:
            noisy_points.append(point)
            
    # Add pure false positives
    num_false_positives = int(len(points) * false_positive_rate)
    for _ in range(num_false_positives):
        # random point within perceptual range
        r = perceptual_range * np.sqrt(random.random())
        theta = random.random() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        noisy_points.append([x, y])

    return noisy_points

def augment_points(points, rotation_angle=15, scale_range=0.1, translation_range=1.0):
    points_arr = np.array(points)
    if points_arr.shape[0] == 0:
        return []

    # Rotation
    angle = np.radians(np.random.uniform(-rotation_angle, rotation_angle))
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    points_arr = points_arr @ rotation_matrix.T

    # Scaling
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    points_arr = points_arr * scale

    # Translation
    translation = np.random.uniform(-translation_range, translation_range, size=2)
    points_arr = points_arr + translation
    
    return points_arr.tolist()

# 3. Create custom dataset class
class LaneDetectionDataset(Dataset):
    def __init__(self, perceptual_field_data, augment=False):
        self.data = perceptual_field_data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_left, noisy_right = self.data[idx]
        
        if self.augment:
            # Augment both left and right boundaries together to maintain their spatial relationship
            combined = np.array(noisy_left + noisy_right)
            augmented_combined = augment_points(combined)
            
            # Split them back
            len_left = len(noisy_left)
            noisy_left = augmented_combined[:len_left]
            noisy_right = augmented_combined[len_left:]

        left_tensor = torch.tensor(noisy_left, dtype=torch.float32)
        right_tensor = torch.tensor(noisy_right, dtype=torch.float32)
        return left_tensor, right_tensor

# 4. Define the model architecture with regularization
class ConeClassifier(nn.Module):
    def __init__(self):
        super(ConeClassifier, self).__init__()
        # Convolutional layers with reduced complexity
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Adaptive pooling to get fixed size output regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)  # Fixed output size
        
        # Calculate the flattened size after conv layers and pooling
        self._get_conv_output = lambda x: 32 * 16  # channels * fixed_length
        
        # Fully connected layers with reduced complexity
        self.fc1 = nn.Linear(self._get_conv_output(None), 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Ensure input is 3D: [batch_size, sequence_length, channels]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Change to [batch, channels, sequence_length]
        x = x.permute(0, 2, 1)  # [batch, 2, points]
        
        # Convolutional layers with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
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

def train_model(train_dataset, val_dataset, model, epochs=50, batch_size=128, learning_rate=0.001):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Add weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_no_improve = 0
    patience = 50  # Reduced patience for small dataset
    min_delta = 0.0001  # Minimum change to qualify as improvement
    
    print(f"Starting training with early stopping patience: {patience} epochs")
    print(f"Minimum improvement threshold: {min_delta}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for left_tensor, right_tensor in train_dataloader:
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
        
        epoch_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct / total
        
        # --- Validation Step ---
        val_loss, val_accuracy = evaluate_model(model, val_dataloader)
        
        # Update learning rate based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log learning rate changes
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Improved early stopping: check both loss and accuracy with minimum delta
        improvement = False
        
        # Check validation loss improvement
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            improvement = True
            print(f"New best validation loss: {val_loss:.4f}")
        
        # Check validation accuracy improvement
        if val_accuracy > (best_val_accuracy + min_delta):
            best_val_accuracy = val_accuracy
            improvement = True
            print(f"New best validation accuracy: {val_accuracy:.2f}%")
        
        # Save best model if either loss or accuracy improved significantly
        if improvement:
            print(f"Saving best model to best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_accuracy,
                'loss': val_loss,
            }, 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_accuracy:.2f}%")
            break

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for left_tensor, right_tensor in dataloader:
            batch_size = left_tensor.size(0)

            # Create labels
            labels_left = torch.zeros(batch_size, dtype=torch.long)
            labels_right = torch.ones(batch_size, dtype=torch.long)

            # Get predictions
            left_pred = model(left_tensor)
            right_pred = model(right_tensor)

            # Calculate loss
            loss_left = criterion(left_pred, labels_left)
            loss_right = criterion(right_pred, labels_right)
            running_loss += (loss_left + loss_right).item()
            
            # Calculate accuracy
            _, predicted_left = torch.max(left_pred.data, 1)
            _, predicted_right = torch.max(right_pred.data, 1)
            
            # Count correct predictions
            total += batch_size * 2
            correct += (predicted_left == labels_left).sum().item()
            correct += (predicted_right == labels_right).sum().item()
    
    accuracy = 100 * correct / total
    loss = running_loss / len(dataloader)
    return loss, accuracy

def load_model(model_path='model.pth'):
    """Load a pre-trained model from file"""
    model = ConeClassifier()
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different save formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'accuracy' in checkpoint:
                print(f"Model loaded from {model_path}")
                print(f"Best validation accuracy: {checkpoint['accuracy']:.2f}%")
                print(f"Best validation loss: {checkpoint['loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded successfully from {model_path}")
            
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main(mode='train', model_path='model.pth'):
    """
    Main function with mode selection
    mode: 'train' to train a new model, 'eval' to only evaluate existing model
    model_path: path to the saved model file
    """
    perceptual_field_data = generate_perceptual_field_data(boundaries, cone_maps)
    
    # --- Create Train/Validation Split ---
    full_dataset = LaneDetectionDataset(perceptual_field_data)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))  # 80% train, 20% validation
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Create train and validation datasets with augmentation for training
    train_dataset = LaneDetectionDataset([(full_dataset.data[i]) for i in train_indices], augment=True)
    val_dataset = LaneDetectionDataset([(full_dataset.data[i]) for i in val_indices], augment=False)  # No augmentation on validation

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    if mode == 'eval':
        # Only evaluate existing model
        model = load_model(model_path)
        if model is not None:
            print("Evaluating on validation set...")
            _, accuracy = evaluate_model(model, val_loader)
            print(f"Test Accuracy: {accuracy:.2f}%")
        else:
            print("Cannot evaluate: model loading failed")
    else:
        # Train new model (default behavior)
        model = ConeClassifier()
        train_model(train_dataset, val_dataset, model)
        
        # Load the best model and evaluate
        print("Loading best model for final evaluation...")
        best_model = load_model('best_model.pth')
        if best_model:
            _, final_accuracy = evaluate_model(best_model, val_loader)
            print(f"Final validation accuracy of best model: {final_accuracy:.2f}%")
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_config': {
                'input_size': 2,
                'conv1_channels': 16,
                'conv2_channels': 32,
                'pool_size': 16,
                'fc1_size': 64,
                'output_size': 2
            }
        }, 'model.pth')

if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'
        main(mode, model_path)
    else:
        # Default behavior - you can change this to 'eval' if you want
        main('train')