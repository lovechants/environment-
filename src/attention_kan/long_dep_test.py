from os import name
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import math
import torch.nn.functional as F
from matplotlib import gridspec

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate synthetic event sequence data
def generate_event_sequence_data(num_samples=1000, seq_length=10, num_features=5, num_classes=3):
    """Generate synthetic event sequence data with temporal patterns."""
    X = np.zeros((num_samples, seq_length, num_features))
    y = np.zeros(num_samples, dtype=np.int64)
    
    for i in range(num_samples):
        # Generate random sequence with temporal patterns
        pattern_type = np.random.randint(0, num_classes)
        y[i] = pattern_type
        
        # Base sequence
        X[i] = np.random.randn(seq_length, num_features) * 0.1
        
        if pattern_type == 0:
            # Increasing trend in first feature
            X[i, :, 0] += np.linspace(0, 1, seq_length)
        elif pattern_type == 1:
            # Periodic pattern in second feature
            X[i, :, 1] += np.sin(np.linspace(0, 3*np.pi, seq_length))
        else:
            # Spike pattern in third feature
            spike_pos = np.random.randint(2, seq_length-2)
            X[i, spike_pos-1:spike_pos+2, 2] += 1.0
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    return X_tensor, y_tensor

def generate_complex_event_sequence_data(num_samples=1000, seq_length=20, num_features=8, num_classes=5, noise_level=0.2):
    """
    Generate complex synthetic event sequence data with multiple temporal patterns,
    varying intensities, and realistic noise.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    seq_length : int
        Length of each sequence
    num_features : int
        Number of features in each sequence
    num_classes : int
        Number of different pattern classes to generate
    noise_level : float
        Amount of noise to add to the sequences
        
    Returns:
    --------
    X : torch.FloatTensor
        Tensor of shape (num_samples, seq_length, num_features)
    y : torch.LongTensor
        Tensor of shape (num_samples,) containing class labels
    """
    X = np.zeros((num_samples, seq_length, num_features))
    y = np.zeros(num_samples, dtype=np.int64)
    
    # Time steps for the sequence
    t = np.linspace(0, 1, seq_length)
    
    for i in range(num_samples):
        # Base noise for all features
        X[i] = np.random.randn(seq_length, num_features) * noise_level
        
        # Assign a pattern class
        pattern_type = np.random.randint(0, num_classes)
        y[i] = pattern_type
        
        # Pattern intensity (makes the task harder by varying the signal strength)
        intensity = np.random.uniform(0.8, 1.5)
        
        # Apply different patterns based on class
        if pattern_type == 0:
            # Class 0: Increasing trend in feature 0, decreasing in feature 1
            X[i, :, 0] += np.linspace(0, 1, seq_length) * intensity
            X[i, :, 1] += np.linspace(1, 0, seq_length) * intensity * 0.7
            # Add a small sinusoidal component to feature 2
            X[i, :, 2] += np.sin(2 * np.pi * 3 * t) * 0.3
            
        elif pattern_type == 1:
            # Class 1: Sinusoidal pattern with different frequencies
            X[i, :, 0] += np.sin(2 * np.pi * 2 * t) * intensity
            X[i, :, 1] += np.cos(2 * np.pi * 2 * t) * intensity * 0.8
            # Add a small linear trend to feature 3
            X[i, :, 3] += np.linspace(0, 0.5, seq_length)
            
        elif pattern_type == 2:
            # Class 2: Spike pattern at random positions
            spike_pos1 = np.random.randint(seq_length // 4, seq_length // 2)
            spike_pos2 = np.random.randint(seq_length // 2, 3 * seq_length // 4)
            
            # Create Gaussian spikes
            spike1 = np.exp(-0.5 * ((np.arange(seq_length) - spike_pos1) / 2) ** 2)
            spike2 = np.exp(-0.5 * ((np.arange(seq_length) - spike_pos2) / 2) ** 2)
            
            X[i, :, 2] += spike1 * intensity
            X[i, :, 3] += spike2 * intensity * 0.7
            # Add a small oscillation to feature 1
            X[i, :, 1] += np.sin(2 * np.pi * 1 * t) * 0.2
            
        elif pattern_type == 3:
            # Class 3: Step function with plateau
            step_pos = np.random.randint(seq_length // 4, 3 * seq_length // 4)
            X[i, :step_pos, 4] += 0.2 * intensity
            X[i, step_pos:, 4] += 1.0 * intensity
            # Add a decreasing component to feature 5
            X[i, :, 5] += np.linspace(0.8, 0.2, seq_length) * intensity * 0.6
            
        else:  # pattern_type == 4
            # Class 4: Exponential growth and decay
            mid_point = seq_length // 2
            # Growth phase
            X[i, :mid_point, 6] += np.exp(np.linspace(0, 2, mid_point)) * intensity * 0.3
            # Decay phase
            X[i, mid_point:, 6] += np.exp(np.linspace(2, 0, seq_length - mid_point)) * intensity * 0.3
            # Add oscillation to feature 7
            X[i, :, 7] += np.sin(2 * np.pi * 4 * t) * intensity * 0.4
            
        # Add cross-feature correlations to make it more challenging
        if np.random.rand() > 0.5:
            # Randomly add correlation between two features
            f1, f2 = np.random.choice(num_features, 2, replace=False)
            correlation_strength = np.random.uniform(0.3, 0.6)
            X[i, :, f2] += X[i, :, f1] * correlation_strength
        
        # Add temporal anomalies (brief deviations from the pattern)
        if np.random.rand() > 0.7:
            # Random anomaly in a random feature
            anomaly_feature = np.random.randint(0, num_features)
            anomaly_start = np.random.randint(0, seq_length - 3)
            anomaly_length = np.random.randint(1, 4)
            X[i, anomaly_start:anomaly_start+anomaly_length, anomaly_feature] += np.random.randn() * 2
    
    # Normalize each feature to have similar scale
    for f in range(num_features):
        feature_std = np.std(X[:, :, f])
        if feature_std > 0:
            X[:, :, f] = X[:, :, f] / feature_std
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    return X_tensor, y_tensor



def generate_challenging_event_sequences(num_samples=2000, seq_length=30, num_features=10, 
                                         num_classes=5, noise_level=0.3):
    """
    Generate challenging synthetic event sequence data with properties that stress model capabilities.
    
    Includes:
    - Long-range dependencies
    - Variable-length patterns
    - Concept drift
    - Rare events
    - Mixed temporal scales
    - Multivariate correlations
    - Class imbalance
    - Adversarial examples
    """
    X = np.zeros((num_samples, seq_length, num_features))
    y = np.zeros(num_samples, dtype=np.int64)
    
    # Create class imbalance (80% in first 3 classes, 20% in remaining classes)
    class_probs = np.ones(num_classes)
    if num_classes > 3:
        class_probs[:3] = 4 * class_probs[3:].sum() / 3
        class_probs = class_probs / class_probs.sum()
    
    # Time steps for the sequence
    t = np.linspace(0, 1, seq_length)
    
    for i in range(num_samples):
        # Base noise for all features
        X[i] = np.random.randn(seq_length, num_features) * noise_level
        
        # Assign a pattern class with imbalance
        pattern_type = np.random.choice(num_classes, p=class_probs)
        y[i] = pattern_type
        
        # Pattern intensity (makes the task harder by varying the signal strength)
        intensity = np.random.uniform(0.6, 1.5)
        
        # Add concept drift - patterns evolve over time
        drift_factor = np.random.uniform(0.7, 1.3)
        
        # Determine if this should be an adversarial example (5% chance)
        is_adversarial = np.random.random() < 0.05
        
        # Apply different patterns based on class
        if pattern_type == 0:
            # Class 0: Long-range dependency - early pattern determines late pattern
            early_pattern = np.random.uniform(-1, 1)
            X[i, :seq_length//4, 0] += early_pattern * intensity
            
            # The late pattern depends on the early pattern
            # Fix: Calculate exact lengths to avoid broadcasting issues
            late_start = 3*seq_length//4
            late_length = seq_length - late_start
            
            if early_pattern > 0:
                X[i, late_start:, 1] += np.linspace(0, 1, late_length) * intensity * drift_factor
            else:
                X[i, late_start:, 2] += np.linspace(1, 0, late_length) * intensity * drift_factor
                
            # Add a misleading pattern if adversarial
            if is_adversarial:
                mid_start = seq_length//3
                mid_length = 2*seq_length//3 - mid_start
                X[i, mid_start:2*seq_length//3, 3] += np.sin(2 * np.pi * 3 * t[mid_start:2*seq_length//3]) * intensity
            
        elif pattern_type == 1:
            # Class 1: Multi-scale temporal patterns
            # Fast oscillation
            X[i, :, 0] += np.sin(2 * np.pi * 8 * t) * intensity * 0.7
            # Medium oscillation
            X[i, :, 1] += np.sin(2 * np.pi * 3 * t) * intensity * 0.8
            # Slow trend
            X[i, :, 2] += np.linspace(-0.5, 0.5, seq_length) * intensity * drift_factor
            
            # Add a rare event - sudden spike
            if np.random.random() < 0.3:  # Only 30% of class 1 has this
                spike_pos = np.random.randint(seq_length//2, seq_length-2)
                X[i, spike_pos:spike_pos+2, 3] += 2.0 * intensity
            
        elif pattern_type == 2:
            # Class 2: Variable-length patterns
            pattern_length = np.random.randint(seq_length//6, seq_length//3)
            start_pos = np.random.randint(0, seq_length - pattern_length)
            
            # Create a variable-length pattern
            pattern = np.sin(np.linspace(0, np.random.randint(1, 4) * np.pi, pattern_length))
            X[i, start_pos:start_pos+pattern_length, 0] += pattern * intensity
            
            # Add correlated features with lag
            lag = np.random.randint(1, 5)
            if start_pos + pattern_length + lag < seq_length:
                X[i, start_pos+lag:start_pos+pattern_length+lag, 1] += pattern * intensity * 0.7 * drift_factor
            
        elif pattern_type == 3:
            # Class 3: Multivariate correlations with changing relationships
            # Generate two base signals
            signal1 = np.random.randn(seq_length) * 0.2 + np.linspace(0, 1, seq_length)
            signal2 = np.random.randn(seq_length) * 0.2 + np.sin(2 * np.pi * 2 * t)
            
            # First half: positive correlation
            X[i, :seq_length//2, 0] += signal1[:seq_length//2] * intensity
            X[i, :seq_length//2, 1] += signal1[:seq_length//2] * 0.8 * intensity
            
            # Second half: negative correlation
            X[i, seq_length//2:, 0] += signal2[seq_length//2:] * intensity * drift_factor
            X[i, seq_length//2:, 1] -= signal2[seq_length//2:] * 0.8 * intensity * drift_factor
            
            # Add a third feature with complex relationship
            X[i, :, 2] += (signal1 * signal2) * intensity * 0.5
            
        else:  # pattern_type >= 4
            # Classes 4+: Mixture of patterns with phase shifts and rare events
            # Base oscillation with random phase shift
            phase = np.random.uniform(0, 2*np.pi)
            X[i, :, 0] += np.sin(2 * np.pi * 2 * t + phase) * intensity
            
            # Add trend with random direction
            trend_dir = np.random.choice([-1, 1])
            X[i, :, 1] += np.linspace(0, trend_dir, seq_length) * intensity * drift_factor
            
            # Add rare events (sudden changes)
            num_events = np.random.randint(1, 4)
            for _ in range(num_events):
                event_pos = np.random.randint(0, seq_length)
                event_width = np.random.randint(1, 4)
                event_feature = np.random.randint(2, min(5, num_features))
                
                if event_pos + event_width < seq_length:
                    X[i, event_pos:event_pos+event_width, event_feature] += np.random.uniform(1.5, 3.0) * intensity
        
        # Add cross-feature correlations with non-linear relationships
        if np.random.rand() > 0.3:  # 70% chance
            f1, f2, f3 = np.random.choice(num_features, 3, replace=False)
            # Non-linear relationship
            X[i, :, f3] += np.tanh(X[i, :, f1] * X[i, :, f2]) * intensity * 0.7
        
        # Add temporal masking (missing data) to random segments
        if np.random.rand() > 0.7:  # 30% chance
            mask_start = np.random.randint(0, seq_length - seq_length//10)
            mask_length = np.random.randint(seq_length//10, seq_length//5)
            mask_feature = np.random.randint(0, num_features)
            
            if mask_start + mask_length < seq_length:
                X[i, mask_start:mask_start+mask_length, mask_feature] = 0
    
    # Add global noise that increases over time (simulating sensor degradation)
    time_dependent_noise = np.linspace(0, 1, seq_length).reshape(1, seq_length, 1) * np.random.randn(num_samples, seq_length, num_features) * noise_level * 0.5
    X += time_dependent_noise
    
    # Normalize each feature to have similar scale
    for f in range(num_features):
        feature_std = np.std(X[:, :, f])
        if feature_std > 0:
            X[:, :, f] = X[:, :, f] / feature_std
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    return X_tensor, y_tensor

class BaseKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=2):
        super(BaseKAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        
        # Projection layer to map flattened input to hidden dimension
        # The input dimension should be input_dim * seq_length
        self.projection = nn.Linear(input_dim * seq_length, hidden_dim)
        
        # KAN layers
        self.kan_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.kan_layers.append(KANLayer(hidden_dim))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Flatten the sequence dimension and feature dimension
        x = x.reshape(batch_size, -1)  # [batch_size, seq_length * input_dim]
        
        # Project to hidden dimension
        x = self.projection(x)
        
        # Apply KAN layers
        for layer in self.kan_layers:
            x = layer(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


class KANLayer(nn.Module):
    def __init__(self, hidden_dim, num_basis=8):
        super(KANLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        
        # Learnable weights for the basis functions
        self.weights = nn.Parameter(torch.randn(hidden_dim, num_basis))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Learnable parameters for the basis functions
        self.a = nn.Parameter(torch.randn(hidden_dim, num_basis))
        self.b = nn.Parameter(torch.randn(hidden_dim, num_basis))
        
    def forward(self, x):
        # x shape: [batch_size, hidden_dim]
        batch_size = x.size(0)
        
        # Expand x for basis function computation
        # [batch_size, hidden_dim, 1]
        x_expanded = x.unsqueeze(2)
        
        # Compute basis functions: sin(a*x + b)
        # [batch_size, hidden_dim, num_basis]
        basis = torch.sin(self.a.unsqueeze(0) * x_expanded + self.b.unsqueeze(0))
        
        # Apply weights to basis functions and sum
        # [batch_size, hidden_dim]
        out = torch.sum(self.weights.unsqueeze(0) * basis, dim=2) + self.bias
        
        return out



class KANWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=2, 
                 num_heads=4, dropout=0.1):
        super(KANWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        
        # Ensure embed_dim is divisible by num_heads
        # Project input to a dimension that's divisible by num_heads
        self.embed_dim = num_heads * ((input_dim + num_heads - 1) // num_heads)  # Round up to nearest multiple
        self.input_projection = nn.Linear(input_dim, self.embed_dim)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,  # Now guaranteed to be divisible by num_heads
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer
        self.projection = nn.Linear(self.embed_dim, hidden_dim)
        
        # KAN layers
        self.kan_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.kan_layers.append(KANLayer(hidden_dim))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        
        # Project input to embed_dim
        x = self.input_projection(x)  # [batch_size, seq_length, embed_dim]
        
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, embed_dim]
        
        # Project to hidden dimension
        x = self.projection(x)
        
        # Apply KAN layers
        for layer in self.kan_layers:
            x = layer(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x

class KANWithLatentAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=2, 
                 num_heads=4, dropout=0.1):
        super(KANWithLatentAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        
        # Projection layer to map input to hidden dimension
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # KAN layers with latent attention
        self.kan_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.kan_layers.append(KANLayerWithLatentAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Process each time step through the network
        outputs = []
        for t in range(self.seq_length):
            # Get current time step
            x_t = x[:, t, :]
            
            # Project to hidden dimension
            h_t = self.projection(x_t)
            
            # Apply KAN layers with latent attention
            for layer in self.kan_layers:
                h_t = layer(h_t)
            
            outputs.append(h_t)
        
        # Stack outputs and pool across time
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_length, hidden_dim]
        pooled = torch.mean(outputs, dim=1)    # [batch_size, hidden_dim]
        
        # Output layer
        out = self.fc_out(pooled)
        
        return out



class KANLayerWithLatentAttention(nn.Module):
    def __init__(self, hidden_dim, num_basis=8, num_heads=4, dropout=0.1):
        super(KANLayerWithLatentAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        
        # Learnable weights for the basis functions
        self.weights = nn.Parameter(torch.randn(hidden_dim, num_basis))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Learnable parameters for the basis functions
        self.a = nn.Parameter(torch.randn(hidden_dim, num_basis))
        self.b = nn.Parameter(torch.randn(hidden_dim, num_basis))
        
        # Latent attention mechanism
        # Ensure latent_dim is divisible by num_heads
        self.latent_dim = num_heads * ((hidden_dim // 2 + num_heads - 1) // num_heads)  # Round up
        self.query_proj = nn.Linear(hidden_dim, self.latent_dim)
        self.key_proj = nn.Linear(hidden_dim, self.latent_dim)
        self.value_proj = nn.Linear(hidden_dim, self.latent_dim)
        
        self.mha = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, hidden_dim]
        batch_size = x.size(0)
        
        # Expand x for basis function computation
        # [batch_size, hidden_dim, 1]
        x_expanded = x.unsqueeze(2)
        
        # Compute basis functions: sin(a*x + b)
        # [batch_size, hidden_dim, num_basis]
        basis = torch.sin(self.a.unsqueeze(0) * x_expanded + self.b.unsqueeze(0))
        
        # Apply weights to basis functions and sum
        # [batch_size, hidden_dim]
        kan_out = torch.sum(self.weights.unsqueeze(0) * basis, dim=2) + self.bias
        
        # Apply latent attention
        # Project to latent space
        q = self.query_proj(kan_out).unsqueeze(1)  # [batch_size, 1, latent_dim]
        k = self.key_proj(kan_out).unsqueeze(1)    # [batch_size, 1, latent_dim]
        v = self.value_proj(kan_out).unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        # Apply attention
        attn_out, _ = self.mha(q, k, v)
        attn_out = attn_out.squeeze(1)  # [batch_size, latent_dim]
        
        # Project back to original dimension
        attn_out = self.output_proj(attn_out)
        
        # Residual connection and normalization
        out = x + self.dropout(attn_out)
        out = self.norm(out)
        
        return out


class KANWithFourier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=2, 
                 num_freqs=8):
        super(KANWithFourier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_freqs = num_freqs
        
        # Fourier feature mapping
        self.register_buffer('freqs', torch.randn(num_freqs) * 2)
        
        # Expanded input dimension after Fourier features
        expanded_dim = input_dim * (2 * num_freqs + 1)
        
        # Projection layer
        self.projection = nn.Linear(expanded_dim, hidden_dim)
        
        # KAN layers
        self.kan_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.kan_layers.append(KANLayer(hidden_dim))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def fourier_features(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # Expand x for frequency computation
        # [batch_size, input_dim, 1]
        x_expanded = x.unsqueeze(2)
        
        # Compute Fourier features: [sin(freq*x), cos(freq*x)]
        # [batch_size, input_dim, num_freqs]
        sin_features = torch.sin(x_expanded * self.freqs)
        cos_features = torch.cos(x_expanded * self.freqs)
        
        # Concatenate original features, sin features, and cos features
        # Reshape to [batch_size, input_dim * (2*num_freqs + 1)]
        sin_features = sin_features.reshape(batch_size, -1)
        cos_features = cos_features.reshape(batch_size, -1)
        
        features = torch.cat([x, sin_features, cos_features], dim=1)
        
        return features
    
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, input_dim]
        
        # Apply Fourier feature mapping
        x = self.fourier_features(x)
        
        # Project to hidden dimension
        x = self.projection(x)
        
        # Apply KAN layers
        for layer in self.kan_layers:
            x = layer(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


class KANWithFourierAndMLA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, num_layers=2, 
                 num_freqs=8, num_heads=4, dropout=0.1):
        super(KANWithFourierAndMLA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_freqs = num_freqs
        
        # Fourier feature mapping
        self.register_buffer('freqs', torch.randn(num_freqs) * 2)
        
        # Expanded input dimension after Fourier features
        expanded_dim = input_dim * (2 * num_freqs + 1)
        
        # Projection layer
        self.projection = nn.Linear(expanded_dim, hidden_dim)
        
        # KAN layers with latent attention
        self.kan_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.kan_layers.append(KANLayerWithLatentAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def fourier_features(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # Expand x for frequency computation
        # [batch_size, input_dim, 1]
        x_expanded = x.unsqueeze(2)
        
        # Compute Fourier features: [sin(freq*x), cos(freq*x)]
        # [batch_size, input_dim, num_freqs]
        sin_features = torch.sin(x_expanded * self.freqs)
        cos_features = torch.cos(x_expanded * self.freqs)
        
        # Concatenate original features, sin features, and cos features
        # Reshape to [batch_size, input_dim * (2*num_freqs + 1)]
        sin_features = sin_features.reshape(batch_size, -1)
        cos_features = cos_features.reshape(batch_size, -1)
        
        features = torch.cat([x, sin_features, cos_features], dim=1)
        
        return features
    
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # [batch_size, input_dim]
        
        # Apply Fourier feature mapping
        x = self.fourier_features(x)
        
        # Project to hidden dimension
        x = self.projection(x)
        
        # Apply KAN layers with latent attention
        for layer in self.kan_layers:
            x = layer(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


# Baseline model: LSTM-based sequence classifier
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(LSTMBaseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the final hidden state from both directions
        final_hidden = lstm_out[:, -1, :]
        
        # Output layer
        out = self.fc_out(final_hidden)
        
        return out


# Markov Chain Baseline
class MarkovChainBaseline:
    def __init__(self, num_classes, seq_length, num_features):
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_features = num_features
        self.transition_matrices = None
        self.initial_probs = None
        self.feature_means = None
        self.feature_stds = None
        
    def fit(self, X, y):
        """
        Fit Markov Chain model to the data.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input sequences of shape [batch_size, seq_length, num_features]
        y : torch.Tensor
            Class labels of shape [batch_size]
        """
        X_np = X.numpy()
        y_np = y.numpy()
        
        # Initialize parameters
        self.transition_matrices = []
        self.initial_probs = []
        self.feature_means = []
        self.feature_stds = []
        
        # Discretize the continuous features for each class
        for c in range(self.num_classes):
            # Get sequences for this class
            class_indices = np.where(y_np == c)[0]
            class_sequences = X_np[class_indices]
            
            # Calculate feature statistics
            means = np.mean(class_sequences, axis=(0, 1))
            stds = np.std(class_sequences, axis=(0, 1))
            self.feature_means.append(means)
            self.feature_stds.append(stds)
            
            # Discretize features (above/below mean)
            discretized = (class_sequences > means.reshape(1, 1, -1)).astype(int)
            
            # Calculate initial state probabilities
            initial_states = discretized[:, 0, :]
            initial_prob = np.mean(initial_states, axis=0)
            self.initial_probs.append(initial_prob)
            
            # Calculate transition matrices for each feature
            feature_transitions = []
            for f in range(self.num_features):
                # Count transitions from 0->0, 0->1, 1->0, 1->1
                transitions = np.zeros((2, 2))
                
                for seq in discretized:
                    for t in range(self.seq_length - 1):
                        from_state = seq[t, f]
                        to_state = seq[t+1, f]
                        transitions[from_state, to_state] += 1
                
                # Normalize to get probabilities
                row_sums = transitions.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1
                transitions = transitions / row_sums
                
                feature_transitions.append(transitions)
            
            self.transition_matrices.append(feature_transitions)
    
    def predict(self, X):
        """
        Predict class labels for input sequences.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input sequences of shape [batch_size, seq_length, num_features]
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted class labels of shape [batch_size]
        """
        X_np = X.numpy()
        batch_size = X_np.shape[0]
        
        # Calculate log-likelihood for each sequence under each class model
        log_likelihoods = np.zeros((batch_size, self.num_classes))
        
        for c in range(self.num_classes):
            means = self.feature_means[c]
            discretized = (X_np > means.reshape(1, 1, -1)).astype(int)
            
            for i in range(batch_size):
                seq = discretized[i]
                
                # Initial state probability
                log_prob = 0
                for f in range(self.num_features):
                    initial_state = seq[0, f]
                    log_prob += np.log(max(self.initial_probs[c][f] if initial_state == 1 
                                          else (1 - self.initial_probs[c][f]), 1e-10))
                
                # Transition probabilities
                for t in range(self.seq_length - 1):
                    for f in range(self.num_features):
                        from_state = seq[t, f]
                        to_state = seq[t+1, f]
                        trans_prob = self.transition_matrices[c][f][from_state, to_state]
                        log_prob += np.log(max(trans_prob, 1e-10))
                
                log_likelihoods[i, c] = log_prob
        
        # Return the class with highest log-likelihood
        return np.argmax(log_likelihoods, axis=1)


# Model training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5, device=device):
    """
    Train a model with early stopping.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    num_epochs : int
        Maximum number of epochs to train
    patience : int
        Number of epochs to wait for improvement before early stopping
    device : torch.device
        Device to use for training
        
    Returns:
    --------
    model : torch.nn.Module
        Trained model
    history : dict
        Training history
    """
    model = model.to(device)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# Evaluation function
def evaluate_model(model, test_loader, criterion=None, device=device):
    """
    Evaluate a model on test data.
    
    Parameters:
    -----------
    model : torch.nn.Module or MarkovChainBaseline
        The model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    criterion : torch.nn.Module, optional
        Loss function (only used for PyTorch models)
    device : torch.device
        Device to use for evaluation
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    all_preds : numpy.ndarray
        All predictions
    all_labels : numpy.ndarray
        All true labels
    """
    # Check if model is a PyTorch model or Markov Chain
    is_torch_model = isinstance(model, nn.Module)
    
    if is_torch_model:
        model = model.to(device)
        model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    # For PyTorch models
    if is_torch_model:
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # For Markov Chain model
    else:
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            # Get predictions
            predicted = model.predict(inputs)
            
            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class metrics
    class_precision = []
    class_recall = []
    
    for i in range(len(np.unique(all_labels))):
        true_positives = conf_mat[i, i]
        false_positives = conf_mat[:, i].sum() - true_positives
        false_negatives = conf_mat[i, :].sum() - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        class_precision.append(precision)
        class_recall.append(recall)
    
    # Calculate F1 score
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(class_precision, class_recall)]
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_mat,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'f1_scores': f1_scores,
        'macro_f1': np.mean(f1_scores)
    }
    
    if is_torch_model and criterion is not None:
        metrics['test_loss'] = test_loss / len(test_loader.dataset)
    
    return metrics, all_preds, all_labels


# Statistical validation function
def statistical_validation(model_results, alpha=0.05):
    """
    Perform statistical validation to compare model performances.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and lists of accuracy scores as values
    alpha : float
        Significance level for statistical tests
        
    Returns:
    --------
    results : dict
        Statistical test results
    """
    from scipy import stats
    import itertools
    
    results = {
        'mean_accuracies': {},
        'std_accuracies': {},
        'pairwise_ttest': {},
        'significant_differences': {}
    }
    
    # Calculate mean and std for each model
    for model_name, accuracies in model_results.items():
        results['mean_accuracies'][model_name] = np.mean(accuracies)
        results['std_accuracies'][model_name] = np.std(accuracies)
    
    # Perform pairwise t-tests
    model_names = list(model_results.keys())
    for model1, model2 in itertools.combinations(model_names, 2):
        t_stat, p_value = stats.ttest_ind(
            model_results[model1], 
            model_results[model2],
            equal_var=False  # Welch's t-test (doesn't assume equal variances)
        )
        
        results['pairwise_ttest'][(model1, model2)] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha
        }
        
        # Record significant differences
        if p_value < alpha:
            better_model = model1 if results['mean_accuracies'][model1] > results['mean_accuracies'][model2] else model2
            worse_model = model2 if better_model == model1 else model1
            
            results['significant_differences'][(better_model, worse_model)] = {
                'mean_diff': results['mean_accuracies'][better_model] - results['mean_accuracies'][worse_model],
                'p_value': p_value
            }
    
    return results

def evaluate_model_robustness(models, test_loaders, criterion=None, device=device):
    """
    Evaluate models on different test scenarios to assess robustness.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to evaluate
    test_loaders : dict
        Dictionary of test loaders for different scenarios
    criterion : torch.nn.Module, optional
        Loss function
    device : torch.device
        Device to use for evaluation
        
    Returns:
    --------
    robustness_metrics : dict
        Metrics for each model on each test scenario
    """
    robustness_metrics = {name: {} for name in models.keys()}
    
    for scenario, loader in test_loaders.items():
        print(f"\nEvaluating on {scenario} scenario...")
        
        for name, model in models.items():
            print(f"  {name}...")
            
            if isinstance(model, nn.Module):
                metrics, _, _ = evaluate_model(model, loader, criterion, device)
            else:
                metrics, _, _ = evaluate_model(model, loader)
                
            robustness_metrics[name][scenario] = metrics
            
            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
    
    return robustness_metrics



def create_robustness_test_scenarios(X, y, seq_length, num_features):
    """
    Create test scenarios to evaluate model robustness.
    
    Parameters:
    -----------
    X : torch.Tensor
        Input data
    y : torch.Tensor
        Labels
    seq_length : int
        Sequence length
    num_features : int
        Number of features
        
    Returns:
    --------
    test_loaders : dict
        Dictionary of test loaders for different scenarios
    """
    batch_size = 64
    test_loaders = {}
    
    # 1. Standard test set
    indices = torch.randperm(len(X))[:500]
    X_test, y_test = X[indices], y[indices]
    test_loaders['standard'] = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size
    )
    
    # 2. Noisy test set (add Gaussian noise)
    X_noisy = X_test.clone()
    X_noisy += torch.randn_like(X_noisy) * 0.5
    test_loaders['noisy'] = DataLoader(
        TensorDataset(X_noisy, y_test), batch_size=batch_size
    )
    
    # 3. Missing data (randomly mask 20% of values)
    X_missing = X_test.clone()
    mask = torch.rand_like(X_missing) < 0.2
    X_missing[mask] = 0
    test_loaders['missing_data'] = DataLoader(
        TensorDataset(X_missing, y_test), batch_size=batch_size
    )
    
    # 4. Time warping (compress or stretch parts of sequences)
    X_warped = X_test.clone()
    for i in range(len(X_warped)):
        # Choose random segment to warp
        seg_start = np.random.randint(0, seq_length // 2)
        seg_end = np.random.randint(seg_start + seq_length // 4, seq_length)
        seg_length = seg_end - seg_start
        
        # Compress or stretch
        if np.random.random() < 0.5:
            # Compress (repeat some frames)
            repeat_idx = np.random.randint(seg_start, seg_end)
            # Make sure we're assigning slices of the same length
            X_warped[i, seg_start:seg_end] = X_test[i, seg_start:seg_end]
            # Then replace one frame with a repeated frame
            X_warped[i, seg_end-1] = X_test[i, repeat_idx]
        else:
            # Stretch (skip one frame)
            # Make sure we're assigning slices of the same length
            if seg_start + 1 < seg_end:
                X_warped[i, seg_start+1:seg_end] = X_test[i, seg_start:seg_end-1]
                # First frame in segment gets duplicated
                X_warped[i, seg_start] = X_test[i, seg_start]
    
    test_loaders['time_warped'] = DataLoader(
        TensorDataset(X_warped, y_test), batch_size=batch_size
    )
    
    # 5. Feature corruption (completely randomize one feature)
    X_corrupted = X_test.clone()
    corrupt_feature = np.random.randint(0, num_features)
    X_corrupted[:, :, corrupt_feature] = torch.randn(len(X_corrupted), seq_length)
    test_loaders['feature_corrupted'] = DataLoader(
        TensorDataset(X_corrupted, y_test), batch_size=batch_size
    )
    
    return test_loaders

# Visualization functions
def plot_training_history(histories, model_names):
    """
    Plot training and validation metrics for multiple models.
    
    Parameters:
    -----------
    histories : list
        List of training histories
    model_names : list
        List of model names
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[0].plot(history['train_loss'], linestyle='-', label=f'{name} (Train)')
        axes[0].plot(history['val_loss'], linestyle='--', label=f'{name} (Val)')
    
    axes[0].set_title('Loss vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training and validation accuracy
    for i, (history, name) in enumerate(zip(histories, model_names)):
        axes[1].plot(history['train_acc'], linestyle='-', label=f'{name} (Train)')
        axes[1].plot(history['val_acc'], linestyle='--', label=f'{name} (Val)')
    
    axes[1].set_title('Accuracy vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(confusion_matrices, class_names, model_names):
    """
    Plot confusion matrices for multiple models.
    
    Parameters:
    -----------
    confusion_matrices : list
        List of confusion matrices
    class_names : list
        List of class names
    model_names : list
        List of model names
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (cm, name) in enumerate(zip(confusion_matrices, model_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(metrics, model_names):
    """
    Plot performance metrics comparison for multiple models.
    
    Parameters:
    -----------
    metrics : list
        List of metric dictionaries
    model_names : list
        List of model names
    """
    # Extract metrics
    accuracies = [m['accuracy'] for m in metrics]
    macro_f1s = [m['macro_f1'] for m in metrics]
    
    # Class-wise metrics
    class_f1s = [m['f1_scores'] for m in metrics]
    n_classes = len(class_f1s[0])
    class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
    
    # Overall metrics
    ax1 = plt.subplot(gs[0, 0])
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy')
    ax1.bar(x + width/2, macro_f1s, width, label='Macro F1')
    
    ax1.set_title('Overall Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Per-class F1 scores
    ax2 = plt.subplot(gs[0, 1])
    
    x = np.arange(n_classes)
    width = 0.8 / len(model_names)
    
    for i, (f1s, name) in enumerate(zip(class_f1s, model_names)):
        ax2.bar(x + i*width - 0.4 + width/2, f1s, width, label=name)
    
    ax2.set_title('F1 Score by Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision-Recall by class and model
    ax3 = plt.subplot(gs[1, :])
    
    # Prepare data for grouped bar chart
    n_groups = n_classes * len(model_names)
    precision_data = []
    recall_data = []
    group_names = []
    
    for c in range(n_classes):
        for i, (m, name) in enumerate(zip(metrics, model_names)):
            precision_data.append(m['class_precision'][c])
            recall_data.append(m['class_recall'][c])
            group_names.append(f'{name}\nClass {c}')
    
    x = np.arange(n_groups)
    width = 0.35
    
    ax3.bar(x - width/2, precision_data, width, label='Precision')
    ax3.bar(x + width/2, recall_data, width, label='Recall')
    
    ax3.set_title('Precision and Recall by Class and Model')
    ax3.set_xticks(x)
    ax3.set_xticklabels(group_names, rotation=90)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_attention_weights(model, test_loader, device=device):
    """
    Visualize attention weights for a model with attention.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model with attention mechanism
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : torch.device
        Device to use for evaluation
    """
    # Check if model has attention
    has_attention = hasattr(model, 'attention') or any('attention' in name for name, _ in model.named_modules())
    
    if not has_attention:
        print("Model does not have attention mechanism.")
        return
    
    model = model.to(device)
    model.eval()
    
    # Get a batch of data
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    
    # Forward pass with attention weights
    with torch.no_grad():
        if hasattr(model, 'attention'):
            # For models with direct attention attribute
            outputs, attention_weights = model.attention(inputs, inputs, inputs)
        else:
            # For models with attention in submodules
            # This is a simplified approach and might need adjustment based on the model architecture
            outputs = model(inputs)
            
            # Extract attention weights from the model
            # This is model-specific and might need to be adapted
            for name, module in model.named_modules():
                if 'attention' in name and hasattr(module, 'attn_weights'):
                    attention_weights = module.attn_weights
                    break
    
    # Visualize attention weights for a few examples
    n_examples = min(3, inputs.size(0))
    n_heads = attention_weights.size(1) if len(attention_weights.shape) > 3 else 1
    
    fig, axes = plt.subplots(n_examples, n_heads, figsize=(n_heads*3, n_examples*3))
    
    if n_examples == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_examples == 1:
        axes = axes.reshape(1, -1)
    elif n_heads == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_examples):
        for h in range(n_heads):
            if n_heads > 1:
                attn = attention_weights[i, h].cpu().numpy()
            else:
                attn = attention_weights[i].cpu().numpy()
            
            sns.heatmap(attn, cmap='viridis', ax=axes[i, h])
            axes[i, h].set_title(f'Example {i+1}, Head {h+1}')
            axes[i, h].set_xlabel('Key position')
            axes[i, h].set_ylabel('Query position')
    
    plt.tight_layout()
    plt.show()


def visualize_feature_importance(model, test_loader, device=device):
    """
    Visualize feature importance for a model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to analyze
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : torch.device
        Device to use for evaluation
    """
    model = model.to(device)
    model.eval()
    
    # Get a batch of data
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    inputs.requires_grad_(True)
    
    # Forward pass
    outputs = model(inputs)
    
    # Get predicted classes
    _, predicted = torch.max(outputs, 1)
    
    # Compute gradients with respect to inputs
    feature_importances = []
    for i in range(len(predicted)):
        # Zero gradients
        if inputs.grad is not None:
            inputs.grad.zero_()
        
        # Backward pass for the predicted class
        outputs[i, predicted[i]].backward(retain_graph=True)
        
        # Get gradients
        gradients = inputs.grad[i].abs().detach().cpu().numpy()
        
        # Average over batch dimension
        importance = np.mean(gradients, axis=0)
        feature_importances.append(importance)
    
    # Convert to numpy array
    feature_importances = np.array(feature_importances)
    
    # Visualize feature importance for a few examples
    n_examples = min(3, len(feature_importances))
    n_features = feature_importances.shape[1]
    seq_length = feature_importances.shape[0]
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3*n_examples))
    
    if n_examples == 1:
        axes = [axes]
    
    for i in range(n_examples):
        im = axes[i].imshow(feature_importances[i].T, aspect='auto', cmap='viridis')
        axes[i].set_title(f'Example {i+1}, True: {labels[i].item()}, Pred: {predicted[i].item()}')
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('Feature')
        axes[i].set_yticks(range(n_features))
        axes[i].set_yticklabels([f'F{j}' for j in range(n_features)])
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


def visualize_kan_basis_functions(model, device=device):
    """
    Visualize the basis functions learned by a KAN model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        KAN model to analyze
    device : torch.device
        Device to use for evaluation
    """
    model = model.to(device)
    model.eval()
    
    # Check if model has KAN layers
    kan_layers = []
    for name, module in model.named_modules():
        if isinstance(module, KANLayer):
            kan_layers.append((name, module))
    
    if not kan_layers:
        print("Model does not have KAN layers.")
        return
    
    # Visualize basis functions for each KAN layer
    for layer_name, layer in kan_layers:
        # Get parameters
        a = layer.a.detach().cpu().numpy()  # [hidden_dim, num_basis]
        b = layer.b.detach().cpu().numpy()  # [hidden_dim, num_basis]
        weights = layer.weights.detach().cpu().numpy()  # [hidden_dim, num_basis]
        
        # Create input range for visualization
        x = np.linspace(-3, 3, 1000)
        
        # Select a few units to visualize
        n_units = min(4, a.shape[0])
        unit_indices = np.random.choice(a.shape[0], n_units, replace=False)
        
        fig, axes = plt.subplots(n_units, 1, figsize=(10, 3*n_units))
        
        if n_units == 1:
            axes = [axes]
        
        for i, unit_idx in enumerate(unit_indices):
            # Plot individual basis functions
            for j in range(a.shape[1]):
                basis = np.sin(a[unit_idx, j] * x + b[unit_idx, j])
                weighted_basis = weights[unit_idx, j] * basis
                axes[i].plot(x, weighted_basis, alpha=0.5, linestyle='--', 
                           label=f'Basis {j}')
            
            # Plot combined function
            combined = np.zeros_like(x)
            for j in range(a.shape[1]):
                basis = np.sin(a[unit_idx, j] * x + b[unit_idx, j])
                combined += weights[unit_idx, j] * basis
            
            axes[i].plot(x, combined, 'k-', linewidth=2, label='Combined')
            
            axes[i].set_title(f'{layer_name} - Unit {unit_idx}')
            axes[i].set_xlabel('Input')
            axes[i].set_ylabel('Output')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def plot_robustness_comparison(robustness_metrics, model_names):
    """
    Plot robustness comparison across different test scenarios.
    
    Parameters:
    -----------
    robustness_metrics : dict
        Metrics for each model on each test scenario
    model_names : list
        List of model names
    """
    scenarios = list(next(iter(robustness_metrics.values())).keys())
    
    # Extract accuracy and F1 scores
    accuracies = {name: [metrics[scenario]['accuracy'] for scenario in scenarios] 
                 for name, metrics in robustness_metrics.items()}
    f1_scores = {name: [metrics[scenario]['macro_f1'] for scenario in scenarios] 
                for name, metrics in robustness_metrics.items()}
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy comparison
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.8 / len(model_names)
    
    for i, name in enumerate(model_names):
        ax.bar(x + i*width - 0.4 + width/2, accuracies[name], width, label=name)
    
    ax.set_title('Accuracy Across Test Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Plot F1 score comparison
    ax = axes[1]
    
    for i, name in enumerate(model_names):
        ax.bar(x + i*width - 0.4 + width/2, f1_scores[name], width, label=name)
    
    ax.set_title('F1 Score Across Test Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_performance_degradation(robustness_metrics, model_names):
    """
    Plot performance degradation relative to standard test scenario.
    
    Parameters:
    -----------
    robustness_metrics : dict
        Metrics for each model on each test scenario
    model_names : list
        List of model names
    """
    scenarios = list(next(iter(robustness_metrics.values())).keys())
    if 'standard' not in scenarios:
        print("No 'standard' scenario found for comparison")
        return
    
    # Calculate relative performance degradation
    rel_accuracy = {}
    rel_f1 = {}
    
    for name in model_names:
        standard_acc = robustness_metrics[name]['standard']['accuracy']
        standard_f1 = robustness_metrics[name]['standard']['macro_f1']
        
        rel_accuracy[name] = [(robustness_metrics[name][scenario]['accuracy'] / standard_acc) - 1 
                             for scenario in scenarios]
        rel_f1[name] = [(robustness_metrics[name][scenario]['macro_f1'] / standard_f1) - 1 
                       for scenario in scenarios]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot relative accuracy degradation
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.8 / len(model_names)
    
    for i, name in enumerate(model_names):
        ax.bar(x + i*width - 0.4 + width/2, rel_accuracy[name], width, label=name)
    
    ax.set_title('Relative Accuracy Change Across Test Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Relative Change')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(True, alpha=0.3)
    
    # Plot relative F1 degradation
    ax = axes[1]
    
    for i, name in enumerate(model_names):
        ax.bar(x + i*width - 0.4 + width/2, rel_f1[name], width, label=name)
    
    ax.set_title('Relative F1 Score Change Across Test Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Relative Change')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_error_patterns(models, test_loader, class_names, device=device):
    """
    Analyze and visualize error patterns for each model.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to analyze
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    class_names : list
        List of class names
    device : torch.device
        Device to use for evaluation
    """
    # Get predictions from each model
    model_predictions = {}
    true_labels = None
    
    for name, model in models.items():
        print(f"Getting predictions for {name}...")
        
        if isinstance(model, nn.Module):
            model = model.to(device)
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            model_predictions[name] = np.array(all_preds)
            if true_labels is None:
                true_labels = np.array(all_labels)
        else:
            # For Markov Chain model
            all_preds = []
            all_labels = []
            
            for inputs, labels in test_loader:
                predicted = model.predict(inputs)
                
                all_preds.extend(predicted)
                all_labels.extend(labels.numpy())
            
            model_predictions[name] = np.array(all_preds)
            if true_labels is None:
                true_labels = np.array(all_labels)
    
    # Analyze common errors across models
    num_models = len(models)
    num_samples = len(true_labels)
    
    # Count how many models got each sample wrong
    error_counts = np.zeros(num_samples)
    for name in models:
        error_counts += (model_predictions[name] != true_labels).astype(int)
    
    # Identify samples that all models got wrong
    all_wrong = np.where(error_counts == num_models)[0]
    
    # Identify samples that some models got right and others wrong
    some_right = np.where((error_counts > 0) & (error_counts < num_models))[0]
    
    # Identify samples that all models got right
    all_right = np.where(error_counts == 0)[0]
    
    print(f"\nError Analysis:")
    print(f"  Samples all models got wrong: {len(all_wrong)} ({len(all_wrong)/num_samples:.1%})")
    print(f"  Samples some models got right: {len(some_right)} ({len(some_right)/num_samples:.1%})")
    print(f"  Samples all models got right: {len(all_right)} ({len(all_right)/num_samples:.1%})")
    
    # Analyze which classes are most difficult
    class_error_rates = {}
    for name in models:
        class_errors = []
        for c in range(len(class_names)):
            class_mask = (true_labels == c)
            if np.sum(class_mask) > 0:
                error_rate = np.mean(model_predictions[name][class_mask] != true_labels[class_mask])
                class_errors.append(error_rate)
            else:
                class_errors.append(0)
        class_error_rates[name] = class_errors
    
    # Plot class error rates
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.8 / len(models)
    
    for i, name in enumerate(models):
        ax.bar(x + i*width - 0.4 + width/2, class_error_rates[name], width, label=name)
    
    ax.set_title('Error Rate by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Error Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create confusion matrix for each model
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    
    if len(models) == 1:
        axes = [axes]
    
    for i, (name, model) in enumerate(models.items()):
        cm = confusion_matrix(true_labels, model_predictions[name])
        
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[i],
                   xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'{name} Normalized Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()


def visualize_model_agreement(models, test_loader, device=device):
    """
    Visualize agreement between different models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to analyze
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data
    device : torch.device
        Device to use for evaluation
    """
    # Get predictions from each model
    model_predictions = {}
    true_labels = None
    
    for name, model in models.items():
        print(f"Getting predictions for {name}...")
        
        if isinstance(model, nn.Module):
            model = model.to(device)
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            model_predictions[name] = np.array(all_preds)
            if true_labels is None:
                true_labels = np.array(all_labels)
        else:
            # For Markov Chain model
            all_preds = []
            all_labels = []
            
            for inputs, labels in test_loader:
                predicted = model.predict(inputs)
                
                all_preds.extend(predicted)
                all_labels.extend(labels.numpy())
            
            model_predictions[name] = np.array(all_preds)
            if true_labels is None:
                true_labels = np.array(all_labels)
    
    # Calculate agreement matrix
    model_names = list(models.keys())
    num_models = len(model_names)
    agreement_matrix = np.zeros((num_models, num_models))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            agreement = np.mean(model_predictions[name1] == model_predictions[name2])
            agreement_matrix[i, j] = agreement
    
    # Plot agreement matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='viridis',
               xticklabels=model_names, yticklabels=model_names)
    ax.set_title('Model Agreement Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate agreement with ground truth
    accuracy = {name: np.mean(preds == true_labels) for name, preds in model_predictions.items()}
    
    # Plot accuracy vs. average agreement with other models
    avg_agreement = {}
    for i, name in enumerate(model_names):
        # Average agreement with all other models
        avg_agreement[name] = np.mean([agreement_matrix[i, j] for j in range(num_models) if j != i])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name in model_names:
        ax.scatter(accuracy[name], avg_agreement[name], s=100, label=name)
    
    ax.set_xlabel('Accuracy (Agreement with Ground Truth)')
    ax.set_ylabel('Average Agreement with Other Models')
    ax.set_title('Model Accuracy vs. Model Agreement')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations
    for name in model_names:
        ax.annotate(name, (accuracy[name], avg_agreement[name]),
                   xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()



def main():
    # Generate challenging event sequence data
    print("Generating challenging data...")
    X, y = generate_challenging_event_sequences(
        num_samples=3000, 
        seq_length=30, 
        num_features=10, 
        num_classes=5
    )
    
    # Split data into train, validation, and test sets
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    test_size = len(X) - train_size - val_size
    
    indices = torch.randperm(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create robustness test scenarios
    print("Creating robustness test scenarios...")
    test_loaders = create_robustness_test_scenarios(
        X_test, y_test, seq_length=X.shape[1], num_features=X.shape[2]
    )
    test_loaders['standard'] = test_loader  # Add standard test loader
    
    # Model parameters
    input_dim = X.shape[2]  # Number of features
    hidden_dim = 128  # Increased for more complex data
    output_dim = len(torch.unique(y))  # Number of classes
    seq_length = X.shape[1]  # Sequence length
    
    # Initialize models
    print("Initializing models...")
    models = {
        'LSTM': LSTMBaseline(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2),
        'BaseKAN': BaseKAN(input_dim, hidden_dim, output_dim, seq_length, num_layers=3),
        'KAN+Attention': KANWithAttention(input_dim, hidden_dim, output_dim, seq_length, 
                                          num_layers=2, num_heads=4, dropout=0.2),
        'KAN+LatentAttention': KANWithLatentAttention(input_dim, hidden_dim, output_dim, seq_length, 
                                                     num_layers=2, num_heads=4, dropout=0.2),
        'KAN+Fourier': KANWithFourier(input_dim, hidden_dim, output_dim, seq_length, 
                                      num_layers=2, num_freqs=12),
        'KAN+Fourier+MLA': KANWithFourierAndMLA(input_dim, hidden_dim, output_dim, seq_length, 
                                               num_layers=2, num_freqs=12, num_heads=4, dropout=0.2)
    }
    
    # Initialize Markov Chain model
    markov_model = MarkovChainBaseline(output_dim, seq_length, input_dim)
    
    # Train and evaluate models
    criterion = nn.CrossEntropyLoss()
    histories = {}
    trained_models = {}
    
    # Train and evaluate PyTorch models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, patience=7, device=device
        )
        
        histories[name] = history
        trained_models[name] = trained_model
        
    # Train Markov Chain model
    print("\nTraining Markov Chain model...")
    markov_model.fit(X_train, y_train)
    trained_models['MarkovChain'] = markov_model
    
    # Evaluate models on standard test set
    print("\nEvaluating models on standard test set...")
    standard_metrics = {}
    
    for name, model in trained_models.items():
        print(f"  {name}...")
        
        if isinstance(model, nn.Module):
            metrics, preds, labels = evaluate_model(model, test_loader, criterion, device)
        else:
            metrics, preds, labels = evaluate_model(model, test_loader)
            
        standard_metrics[name] = metrics
        
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}")
    
    # Evaluate model robustness across different test scenarios
    print("\nEvaluating model robustness...")
    robustness_metrics = evaluate_model_robustness(trained_models, test_loaders, criterion, device)
    
    # Perform cross-validation for statistical validation
    print("\nPerforming cross-validation for statistical validation...")
    from sklearn.model_selection import KFold
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {name: [] for name in trained_models.keys()}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Split data
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_test_fold, y_test_fold = X[test_idx], y[test_idx]
        
        # Create data loaders
        train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
        test_dataset_fold = TensorDataset(X_test_fold, y_test_fold)
        
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        test_loader_fold = DataLoader(test_dataset_fold, batch_size=batch_size)
        
        # Train and evaluate each model
        for name, model_class in models.items():
            # Initialize a new model
            if name == 'LSTM':
                model = LSTMBaseline(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2)
            elif name == 'BaseKAN':
                model = BaseKAN(input_dim, hidden_dim, output_dim, seq_length, num_layers=3)
            elif name == 'KAN+Attention':
                model = KANWithAttention(input_dim, hidden_dim, output_dim, seq_length, 
                                         num_layers=2, num_heads=4, dropout=0.2)
            elif name == 'KAN+LatentAttention':
                model = KANWithLatentAttention(input_dim, hidden_dim, output_dim, seq_length, 
                                              num_layers=2, num_heads=4, dropout=0.2)
            elif name == 'KAN+Fourier':
                model = KANWithFourier(input_dim, hidden_dim, output_dim, seq_length, 
                                       num_layers=2, num_freqs=12)
            elif name == 'KAN+Fourier+MLA':
                model = KANWithFourierAndMLA(input_dim, hidden_dim, output_dim, seq_length, 
                                            num_layers=2, num_freqs=12, num_heads=4, dropout=0.2)
            
            # Train model with fewer epochs for CV
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            trained_model, _ = train_model(
                model, train_loader_fold, test_loader_fold, criterion, optimizer,
                num_epochs=20, patience=5, device=device
            )
            
            # Evaluate model
            fold_metrics, _, _ = evaluate_model(
                trained_model, test_loader_fold, criterion, device
            )
            
            cv_results[name].append(fold_metrics['accuracy'])
        
        # Train and evaluate Markov Chain model
        markov_model_fold = MarkovChainBaseline(output_dim, seq_length, input_dim)
        markov_model_fold.fit(X_train_fold, y_train_fold)
        
        fold_metrics, _, _ = evaluate_model(
            markov_model_fold, test_loader_fold
        )
        
        cv_results['MarkovChain'].append(fold_metrics['accuracy'])
    
    # Perform statistical validation
    print("\nPerforming statistical validation...")
    stat_results = statistical_validation(cv_results)
    
    # Print statistical validation results
    print("\nMean Accuracies:")
    for model_name, mean_acc in stat_results['mean_accuracies'].items():
        std_acc = stat_results['std_accuracies'][model_name]
        print(f"{model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    print("\nSignificant Differences:")
    for (better, worse), result in stat_results['significant_differences'].items():
        print(f"{better} is significantly better than {worse} "
              f"(diff: {result['mean_diff']:.4f}, p-value: {result['p_value']:.4f})")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(
        [histories[name] for name in models.keys()],
        list(models.keys())
    )
    
    # Plot confusion matrices
    class_names = [f'Class {i}' for i in range(output_dim)]
    plot_confusion_matrices(
        [standard_metrics[name]['confusion_matrix'] for name in trained_models.keys()],
        class_names,
        list(trained_models.keys())
    )
    
    # Plot performance comparison
    plot_performance_comparison(
        [standard_metrics[name] for name in trained_models.keys()],
        list(trained_models.keys())
    )
    
    # Plot robustness comparison
    plot_robustness_comparison(robustness_metrics, list(trained_models.keys()))
    
    # Plot performance degradation
    plot_performance_degradation(robustness_metrics, list(trained_models.keys()))
    
    # Analyze error patterns
    analyze_error_patterns(trained_models, test_loader, class_names, device)
    
    # Visualize model agreement
    visualize_model_agreement(trained_models, test_loader, device)
    
    # Visualize attention weights for models with attention
    for name, model in trained_models.items():
        if isinstance(model, nn.Module) and ('Attention' in name):
            print(f"\nVisualizing attention weights for {name}...")
            visualize_attention_weights(model, test_loader, device)
    
    # Visualize feature importance
    for name, model in trained_models.items():
        if isinstance(model, nn.Module):  # Skip Markov Chain
            print(f"\nVisualizing feature importance for {name}...")
            visualize_feature_importance(model, test_loader, device)
    
    # Visualize KAN basis functions
    for name, model in trained_models.items():
        if isinstance(model, nn.Module) and ('KAN' in name):
            print(f"\nVisualizing basis functions for {name}...")
            visualize_kan_basis_functions(model, device)


if __name__ == "__main__":
    main()

