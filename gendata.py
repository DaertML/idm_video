import gymnasium as gym
import ale_py
import numpy as np
import cv2 # Used for image processing (resizing, grayscaling)

# --- CONFIGURATION ---
ENV_ID = "ALE/SpaceInvaders-v5"
DATASET_SIZE = 10000  # Number of (frame_t, frame_t+1, action) samples to collect
OUTPUT_FILE = "idm_dataset.npy"
TARGET_SHAPE = (84, 84) # Standard Atari frame size

def preprocess_frame(frame):
    """
    Converts a single RGB frame (H, W, 3) to grayscale and resizes it to 84x84.
    The output is normalized to [0, 1].
    
    Args:
        frame (np.ndarray): Input frame from the environment (210, 160, 3).
        
    Returns:
        np.ndarray: Preprocessed frame (84, 84), dtype float32.
    """
    # 1. Convert to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 2. Resize to TARGET_SHAPE (84x84) using INTER_AREA for downsampling
    resized_frame = cv2.resize(gray_frame, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    
    # 3. Normalize pixel values to [0, 1] and ensure float32 type
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    
    return normalized_frame

def collect_idm_data(env_id, num_samples, output_file):
    """
    Collects state transition data (Ft, Ft+1, At) from the environment.
    
    Args:
        env_id (str): Gymnasium environment ID.
        num_samples (int): Total number of data points to collect.
        output_file (str): Filepath to save the collected data.
    """
    print(f"Initializing environment: {env_id}")
    try:
        # Use render_mode='rgb_array' to get the full image observations
        env = gym.make(env_id, render_mode="rgb_array")
    except Exception as e:
        print(f"Error initializing environment. Ensure 'atari' environment dependencies are installed.")
        print("Try running: pip install 'gymnasium[atari]' ale-py opencv-python")
        print(f"Error details: {e}")
        return

    # Action Space: Discrete(6) for Space Invaders (No-op, Fire, Right, Left, Right+Fire, Left+Fire)
    action_space_size = env.action_space.n
    print(f"Action space size: {action_space_size}")
    
    # List to store collected data: (F_t, F_{t+1}, A_t)
    dataset = []
    
    observation, info = env.reset()
    F_t = preprocess_frame(observation)
    
    print(f"Starting data collection for {num_samples} samples...")
    for i in range(num_samples):
        # 1. Sample a random action (A_t)
        A_t = env.action_space.sample()
        
        # 2. Step the environment
        observation_next, reward, terminated, truncated, info = env.step(A_t)
        
        # 3. Preprocess the next frame (F_{t+1})
        F_t_plus_1 = preprocess_frame(observation_next)
        
        # 4. Save the transition
        # Store F_t, F_{t+1}, and the action A_t (as an integer)
        dataset.append({
            'frame_t': F_t,
            'frame_t_plus_1': F_t_plus_1,
            'action': A_t
        })
        
        # 5. F_t becomes F_{t+1} for the next loop iteration
        F_t = F_t_plus_1
        
        # 6. Reset environment if episode ends
        if terminated or truncated:
            observation, info = env.reset()
            F_t = preprocess_frame(observation)
        
        if (i + 1) % 1000 == 0:
            print(f"Collected {i + 1}/{num_samples} samples.")

    env.close()
    
    # Save data using numpy
    np.save(output_file, dataset)
    print(f"\nData collection complete. Saved {len(dataset)} samples to {output_file}")

if __name__ == "__main__":
    # You must have 'gymnasium[atari]' and 'ale-py' installed to run this.
    collect_idm_data(ENV_ID, DATASET_SIZE, OUTPUT_FILE)
