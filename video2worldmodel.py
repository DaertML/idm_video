import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
MODEL_PATH = 'idm_model.pth'
VIDEO_PATH = 'spaceinv_360p.mp4' # Replace with your actual video file path
OUTPUT_DIR = 'predicted_actions_output' # New directory for saving frames
TARGET_SHAPE = (84, 84) # Must match the training size
ACTION_SPACE_SIZE = 6 

# Space Invaders Action Meanings (Discrete(6) default)
ACTION_MEANINGS = [
    "NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"
]

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL ARCHITECTURE (Copied from train_idm.py) ---

class InverseDynamicsModel(nn.Module):
    """
    CNN model to predict the action A_t given two stacked frames (F_t, F_t+1).
    Input shape: (Batch, 2, 84, 84)
    """
    def __init__(self, num_actions):
        super(InverseDynamicsModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_input_size = 7 * 7 * 64
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        action_logits = self.fc_layers(x)
        return action_logits

# --- 2. UTILITY FUNCTIONS ---

def preprocess_frame(frame_rgb):
    """
    Converts an RGB frame to grayscale, resizes it to 84x84, and normalizes.
    This must exactly match the preprocessing in data_collector.py.
    """
    # 1. Convert to Grayscale
    # Note: cv2 reads frames as BGR, so cvtColor expects BGR input
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize
    resized_frame = cv2.resize(gray_frame, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    
    # 3. Normalize
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    
    # Add a channel dimension (C, H, W) for PyTorch conversion later
    return normalized_frame

def load_model(model_path, num_actions):
    """Loads the trained model state dictionary."""
    model = InverseDynamicsModel(num_actions).to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return None
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
        
    return model

# --- 3. VIDEO PREDICTION AND SAVING FUNCTION ---

@torch.no_grad() # Disable gradient calculation for inference
def predict_video_actions(video_path, model):
    """
    Processes a video file frame by frame, predicts the action A_t using pair (F_t, F_t+1).
    For every frame pair, a new sequence folder is created, and the two frames are saved 
    as frame0.png (F_t) and frame1.png (F_t+1, annotated).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}.")
        print("Please ensure you have a valid video file.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Tracking variable for sequence folder creation (increments every prediction step)
    sequence_counter = 0
    
    # Read the first frame (F_t) - This is F_0
    ret, frame_t_rgb = cap.read() # RGB version of F_t (needed for saving)
    if not ret:
        print("Could not read first frame.")
        cap.release()
        return

    # Preprocess F_t immediately
    F_t_preprocessed = preprocess_frame(frame_t_rgb) # Preprocessed version of F_t
    
    frame_count = 0 # This tracks the global index of F_t+1

    print("Starting video prediction. Press 'q' to exit the video player.")
    print(f"Saving output frames to: {os.path.abspath(OUTPUT_DIR)}")

    while True:
        # Read the next frame (F_t+1) - This is F_t+1
        ret, frame_t_plus_1_rgb = cap.read()
        
        # Check if video ended
        if not ret:
            break

        # Sequence counter increments for every single prediction step (i.e., every frame pair)
        sequence_counter += 1
        frame_count += 1 
        
        # 1. Preprocess F_t+1
        F_t_plus_1_preprocessed = preprocess_frame(frame_t_plus_1_rgb)
        
        # 2. Prepare the input tensor for the model
        # Stack F_t and F_t+1: (2, 84, 84)
        input_frames_np = np.stack([F_t_preprocessed, F_t_plus_1_preprocessed], axis=0)
        
        # Add batch dimension and move to device: (1, 2, 84, 84)
        input_tensor = torch.from_numpy(input_frames_np).float().unsqueeze(0).to(DEVICE)

        # 3. Run Inference (Predicting A_t from F_t and F_t+1)
        action_logits = model(input_tensor)
        predicted_action_idx = torch.argmax(action_logits, dim=1).item()
        predicted_action_label = ACTION_MEANINGS[predicted_action_idx]
        
        # 4. Folder Creation (New folder for every pair)
        
        # Define the new folder path for this single frame pair
        sequence_save_dir = os.path.join(
            OUTPUT_DIR, 
            predicted_action_label, 
            f"sequence_{sequence_counter:05d}" # Use 5 digits for larger videos
        )
        # Create the directory structure (e.g., predicted_actions_output/FIRE/sequence_00003)
        os.makedirs(sequence_save_dir, exist_ok=True)
            
        # 5. Display the results and save the annotated pair
        
        # We annotate F_t+1 (the current frame)
        display_frame = frame_t_plus_1_rgb.copy() # Use a copy for annotation
        
        # Overlay the predicted action
        text = f"Predicted Action: {predicted_action_label}"
        cv2.putText(
            display_frame, 
            text, 
            (10, 30), # Position
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, # Font scale
            (0, 255, 0), # Green color (BGR)
            2, # Thickness
            cv2.LINE_AA
        )
        
        # Save F_t (the frame *before* the action happened) as frame0.png
        ft_filename = "frame0.png"
        cv2.imwrite(os.path.join(sequence_save_dir, ft_filename), frame_t_rgb)

        # Save F_t+1 (the frame *after* the action started/completed) (Annotated) as frame1.png
        ft_plus_1_filename = "frame1.png"
        cv2.imwrite(os.path.join(sequence_save_dir, ft_plus_1_filename), display_frame)

        # Display the F_t+1 frame in the viewer
        cv2.imshow('IDM Action Prediction (F_t+1)', display_frame)

        # 6. Advance the frames for the next iteration
        # F_t+1 becomes the new F_t for the next prediction step
        F_t_preprocessed = F_t_plus_1_preprocessed
        frame_t_rgb = frame_t_plus_1_rgb # Crucial: update the RGB reference for F_t saving

        # Wait for key press (1 ms delay for smooth playback)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nVideo processing finished after {sequence_counter} prediction steps.")


if __name__ == '__main__':
    print("--- Inverse Dynamics Model Prediction and Pair Extraction ---")
    
    # 1. Load the model
    model = load_model(MODEL_PATH, ACTION_SPACE_SIZE)
    if model is None:
        exit()
        
    # 2. Run prediction
    # NOTE: You must have an actual video file named 'gameplay_video.mp4' 
    # or change the VIDEO_PATH variable to a valid path.
    predict_video_actions(VIDEO_PATH, model)

