

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_data():
    """Generates training data with a localized high-frequency burst."""
    x = torch.linspace(-np.pi, np.pi, 512).unsqueeze(1)
    y_low_freq = torch.sin(2 * x)
    y_high_freq = torch.sin(40 * x)
    gaussian_window = torch.exp(-x**2 / 0.1)
    y = y_low_freq + y_high_freq * gaussian_window
    return x.to(device), y.to(device)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Applies positional encoding to the input."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Forward pass for the PositionalEncoding."""
        indices = ((x + np.pi) / (2 * np.pi) * (self.pe.shape[1] - 1)).long().squeeze(1)
        return self.pe[:, indices, :].squeeze(0)

# --- MLP Models ---
class SimpleMLP(nn.Module):
    """A simple MLP without positional encoding."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.layers(x)

class PositionalMLP(nn.Module):
    """An MLP that uses positional encoding."""
    def __init__(self, encoding_dim):
        super().__init__()
        self.encoding = PositionalEncoding(d_model=encoding_dim)
        self.layers = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        encoded_x = self.encoding(x)
        return self.layers(encoded_x)


if __name__ == '__main__':
    # --- Setup ---
    epochs = 1500 
    lr = 1e-4
    encoding_dimension = 16
    frame_dir = "training_frames"
    gif_path = "training_evolution.gif"

    # Create directory for frames if it doesn't exist
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # Generate data
    x_train, y_train = generate_data()

    # MLP without positional encoding 
    simple_mlp = SimpleMLP().to(device) 
    # MLP with positional encoding 
    pos_mlp = PositionalMLP(encoding_dim=encoding_dimension).to(device)
    
    optimizer_simple = torch.optim.Adam(simple_mlp.parameters(), lr=lr)
    optimizer_pos = torch.optim.Adam(pos_mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Starting training and generating frames for GIF...")
    # --- Training Loop ---
    for epoch in range(epochs):
        # Train Simple MLP
        optimizer_simple.zero_grad()
        output_simple = simple_mlp(x_train)
        loss_simple = criterion(output_simple, y_train)
        loss_simple.backward()
        optimizer_simple.step()

        # Train Positional MLP
        optimizer_pos.zero_grad()
        output_pos = pos_mlp(x_train)
        loss_pos = criterion(output_pos, y_train)
        loss_pos.backward()
        optimizer_pos.step()

        # --- Generate and Save Frame ---
        # We'll save a frame every 20 epochs to keep the GIF smooth but not too long
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] -> Saving frame')
            
            # Get model predictions
            simple_pred = output_simple.detach().cpu()
            pos_pred = output_pos.detach().cpu()

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.title(f"Training Evolution - Epoch {epoch + 1}")
            plt.plot(x_train.cpu(), y_train.cpu(), label='Ground Truth', color='black', linewidth=2, zorder=10)
            plt.plot(x_train.cpu(), simple_pred, label=f'Simple MLP (Loss: {loss_simple.item():.4f})', color='red', linestyle='--')
            plt.plot(x_train.cpu(), pos_pred, label=f'Positional MLP (Loss: {loss_pos.item():.4f})', color='green', linestyle='--')
            plt.xlabel("Input Coordinate (x)")
            plt.ylabel("Signal Value (y)")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.ylim(-2.5, 2.5) 

            # Save the frame
            frame_path = os.path.join(frame_dir, f"frame_{epoch+1:04d}.png")
            plt.savefig(frame_path)
            plt.close() 

   
    print("\nTraining finished. Compiling frames into a GIF...")
    frames = []
    
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".png")])
    for frame_path in frame_files:
        frames.append(imageio.imread(frame_path))

 
    imageio.mimsave(gif_path, frames, duration=0.2) # 0.1 seconds per frame

    print(f"Successfully created GIF: {gif_path}")




