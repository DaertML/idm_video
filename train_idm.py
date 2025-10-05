import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- CONFIGURATION ---
DATA_FILE = 'idm_dataset.npy'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 1000
ACTION_SPACE_SIZE = 6 # SpaceInvaders default action space

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DATASET AND DATALOADER ---

class IDMDataset(Dataset):
    """PyTorch Dataset for Inverse Dynamics Model training data."""
    def __init__(self, data_file):
        # Load the structured numpy array from the data collector
        print(f"Loading data from {data_file}...")
        raw_data = np.load(data_file, allow_pickle=True)
        
        if len(raw_data) == 0:
            raise ValueError("Dataset is empty. Run data_collector.py first.")

        # Extract frames and actions
        # frames_t and frames_t_plus_1 are (N, 84, 84)
        self.frames_t = np.stack([d['frame_t'] for d in raw_data])
        self.frames_t_plus_1 = np.stack([d['frame_t_plus_1'] for d in raw_data])
        # actions is (N,)
        self.actions = np.array([d['action'] for d in raw_data], dtype=np.int64)
        
        # Ensure data integrity
        assert self.frames_t.shape == self.frames_t_plus_1.shape
        assert len(self.frames_t) == len(self.actions)
        print(f"Loaded {len(self.actions)} samples.")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # Frame shape is (H, W). We add the channel dimension (C=1 for grayscale)
        # and stack them to create the input: (2, 84, 84)
        
        # F_t is (84, 84)
        F_t = self.frames_t[idx]
        # F_t_plus_1 is (84, 84)
        F_t_plus_1 = self.frames_t_plus_1[idx]
        
        # Stack the two frames along the channel dimension (new axis 0)
        # Input tensor will be (C=2, H=84, W=84)
        input_frames = np.stack([F_t, F_t_plus_1], axis=0)
        
        # Convert to PyTorch tensors
        input_frames = torch.from_numpy(input_frames).float()
        action_label = torch.tensor(self.actions[idx]).long() # action must be long for CrossEntropyLoss
        
        return input_frames, action_label

# --- 2. INVERSE DYNAMICS MODEL (IDM) ARCHITECTURE ---

class InverseDynamicsModel(nn.Module):
    """
    CNN model to predict the action A_t given two stacked frames (F_t, F_t+1).
    Based on standard Atari DQN/VPT visual backbones.
    Input shape: (Batch, 2, 84, 84) (2 channels for stacked grayscale frames)
    """
    def __init__(self, num_actions):
        super(InverseDynamicsModel, self).__init__()
        
        # Convolutional Layers (Standard DQN architecture)
        self.conv_layers = nn.Sequential(
            # Input: 2 channels (F_t, F_t+1)
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size of the flattened output to define the MLP head
        # (84, 84) -> Conv8x4 -> (20, 20) -> Conv4x2 -> (9, 9) -> Conv3x1 -> (7, 7)
        # Output size: 64 channels * 7 * 7 = 3136
        self.fc_input_size = 7 * 7 * 64
        
        # MLP Head for Action Classification
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            # Output: number of actions (classification classes)
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        """x is the stacked frames (F_t, F_t+1) with shape (B, 2, 84, 84)"""
        x = self.conv_layers(x)
        action_logits = self.fc_layers(x)
        return action_logits

# --- 3. TRAINING FUNCTION ---

def train_idm(data_file, num_epochs, num_actions):
    """Initializes and trains the Inverse Dynamics Model."""
    
    try:
        dataset = IDMDataset(data_file)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = InverseDynamicsModel(num_actions).to(DEVICE)
    # The IDM task is a classification problem (which action was taken?)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nStarting training on {DEVICE} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_frames, batch_actions in dataloader:
            batch_frames, batch_actions = batch_frames.to(DEVICE), batch_actions.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            action_logits = model(batch_frames)
            
            # Calculate loss
            loss = criterion(action_logits, batch_actions)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_frames.size(0)
            
            # Calculate accuracy
            _, predicted_actions = torch.max(action_logits, 1)
            total_samples += batch_actions.size(0)
            correct_predictions += (predicted_actions == batch_actions).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_accuracy = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    print("Training complete.")
    
    # Save the trained model
    torch.save(model.state_dict(), 'idm_model.pth')
    print("Model saved to idm_model.pth")


if __name__ == '__main__':
    # Ensure you run data_collector.py first to create 'idm_dataset.npy'
    train_idm(DATA_FILE, EPOCHS, ACTION_SPACE_SIZE)
