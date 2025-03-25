"""
Code adapted from, and taken from https://github.com/KindXiaoming/pykan
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import argparse 
import pickle
import os 
from torch.utils.data import DataLoader, Dataset 
from sklearn.model_selection import train_test_split 

def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    '''
    
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda
            
    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    '''
    #print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    #print('mat', mat.shape)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    #print('y_eval', y_eval.shape)
    device = mat.device
    
    #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
    except:
        print('lstsq failed')
    
    # manual psuedo-inverse
    '''lamb=1e-8
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]'''
    
    return coef


def extend_grid(grid, k_extend=0):
    '''
    extend grid
    '''
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid


class KANLayer(nn.Module):
    """
    KANLayer class based on the original implementation
    """
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, 
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, 
                 base_fun=nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], 
                 sp_trainable=True, sb_trainable=True, device='cpu', sparse_init=False):
        super(KANLayer, self).__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.device = device

        # Create grid for splines
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = nn.Parameter(grid).requires_grad_(False)
        
        # Initialize coefficients with noise
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num
        
        # Convert to spline coefficients
        try:
            self.coef = nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        except:
            # Fallback initialization if spline functions are not available
            self.coef = nn.Parameter(torch.randn(self.in_dim, self.num + k, self.out_dim) * noise_scale)
        
        # Masking for sparse networks
        if sparse_init:
            # Create sparse mask if requested
            mask = torch.zeros(in_dim, out_dim)
            # Simple sparse pattern: connect each input to ~sqrt(out_dim) outputs
            connect_count = max(1, int(np.sqrt(out_dim)))
            for i in range(in_dim):
                indices = torch.randperm(out_dim)[:connect_count]
                mask[i, indices] = 1.0
            self.mask = nn.Parameter(mask).requires_grad_(False)
        else:
            self.mask = nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        # Scale parameters
        self.scale_base = nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + 
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)
        
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        
        self.to(device)
        
    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device    
        return self

    def forward(self, x):
        """KANLayer forward pass"""
        batch = x.shape[0]
        preacts = x[:,None,:].clone().expand(batch, self.out_dim, self.in_dim)
            
        # Base function (nonlinearity)
        base = self.base_fun(x)  # (batch, in_dim)
        
        # Spline function
        try:
            y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        except:
            # Fallback computation if spline functions are not available
            # Simple linear interpolation
            y = torch.zeros(batch, self.in_dim, self.out_dim).to(self.device)
            for i in range(self.in_dim):
                for j in range(self.out_dim):
                    # Linear combination of basis
                    basis_values = torch.zeros(batch, self.num + self.k).to(self.device)
                    for b in range(self.num + self.k):
                        # Simple RBF-like basis
                        centers = torch.linspace(-1, 1, self.num + self.k)[b]
                        basis_values[:, b] = torch.exp(-5.0 * (x[:, i] - centers)**2)
                    y[:, i, j] = torch.matmul(basis_values, self.coef[i, :, j])
        
        postspline = y.clone().permute(0,2,1)
            
        # Combine base function and spline with scales
        y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
        
        # Apply mask
        y = self.mask[None,:,:] * y
        
        postacts = y.clone().permute(0,2,1)
        
        # Sum across input dimension
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline


class EventSequenceDataset(Dataset):
    """Dataset for event sequences"""
    def __init__(self, sequences, label_to_idx, seq_length=10):
        self.sequences = sequences
        self.label_to_idx = label_to_idx
        self.seq_length = seq_length
        
        # Process sequences into input/target pairs
        self.inputs = []
        self.targets = []
        
        for seq_id, seq in sequences.items():
            # Convert string labels to indices
            seq_indices = [label_to_idx[label] for label in seq]
            
            # Create training examples from windows
            for i in range(len(seq_indices) - 1):
                # Get prefix of sequence
                end_idx = min(i + 1, len(seq_indices))
                prefix = seq_indices[:end_idx]
                
                # Pad if necessary
                if len(prefix) < self.seq_length:
                    prefix = [0] * (self.seq_length - len(prefix)) + prefix
                else:
                    prefix = prefix[-self.seq_length:]
                
                self.inputs.append(prefix)
                # Target is the next event
                if end_idx < len(seq_indices):
                    self.targets.append(seq_indices[end_idx])
                else:
                    # If we're at the end, use the last event
                    self.targets.append(seq_indices[-1])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


class SequenceEncoder(nn.Module):
    """
    Encodes a sequence of event indices into a continuous representation
    using one-hot encoding and B-spline interpolation
    """
    def __init__(self, num_events, embedding_dim, sequence_length, 
                 num_splines=5, spline_order=3, device='cpu'):
        super(SequenceEncoder, self).__init__()
        self.num_events = num_events
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.device = device
        
        # Embedding layer (more efficient than one-hot)
        self.embedding = nn.Embedding(num_events, embedding_dim)
        
        # Positional encoding
        position = torch.arange(0, sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                            -(np.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(sequence_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Spline-based position encoding
        # This creates a more flexible positional encoding using B-splines
        self.position_kan = KANLayer(
            in_dim=1,                      # Position is 1D
            out_dim=embedding_dim,         # Project to embedding dimension
            num=num_splines,               # Number of splines
            k=spline_order,                # Order of splines
            scale_base_mu=0.0,             # No base function component
            scale_base_sigma=0.0,          # No base function component
            scale_sp=1.0,                  # Full spline component
            grid_range=[0, 1],             # Normalize positions to [0,1]
            device=device
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length] containing event indices
        
        Returns:
            encoded: Tensor of shape [batch_size, embedding_dim * 2]
                     containing the sequence encoding
        """
        batch_size = x.size(0)
        
        # Embed event indices
        embedded = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        
        # Add standard positional encoding
        embedded = embedded + self.pe.unsqueeze(0)
        
        # Apply KAN-based positional encoding
        # First create normalized positions
        positions = torch.linspace(0, 1, self.sequence_length).unsqueeze(1).to(self.device)
        positions = positions.expand(self.sequence_length, 1)
        
        # Get position encodings from the KAN
        kan_pos_encoding, _, _, _ = self.position_kan(positions)  # [sequence_length, embedding_dim]
        
        # Add KAN-based positional encoding
        embedded = embedded + kan_pos_encoding.unsqueeze(0)
        
        # Two options for sequence-level representation:
        
        # 1. Attention-weighted pooling
        # (Optional) Simple attention mechanism
        attention_weights = F.softmax(
            torch.matmul(embedded, torch.randn(self.embedding_dim, 1).to(self.device)), 
            dim=1
        )
        attended_features = torch.sum(embedded * attention_weights, dim=1)
        
        # 2. Simple temporal pooling (mean and max)
        mean_features = torch.mean(embedded, dim=1)
        max_features, _ = torch.max(embedded, dim=1)
        
        # Concatenate for final representation
        encoded = torch.cat([attended_features, mean_features, max_features], dim=1)
        
        return encoded


class EventSequenceKAN(nn.Module):
    """
    KAN-based model for event sequence prediction
    Closely follows the original KAN architecture while adapting for sequences
    """
    def __init__(self, num_events, hidden_dim=64, embedding_dim=32, 
                 num_layers=2, sequence_length=10, 
                 num_splines=5, spline_order=3, device='cpu'):
        super(EventSequenceKAN, self).__init__()
        
        self.num_events = num_events
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.device = device
        
        # Encode sequences to continuous representations
        self.encoder = SequenceEncoder(
            num_events=num_events,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            num_splines=num_splines,
            spline_order=spline_order,
            device=device
        )
        
        # Calculate encoder output dimension
        encoder_output_dim = embedding_dim * 3  # attended + mean + max
        
        # KAN layers
        self.kan_layers = nn.ModuleList()
        
        # First layer takes the encoded representation
        self.kan_layers.append(
            KANLayer(
                in_dim=encoder_output_dim,
                out_dim=hidden_dim,
                num=num_splines,
                k=spline_order,
                scale_base_mu=0.0,
                scale_base_sigma=0.2,
                scale_sp=1.0,
                base_fun=nn.SiLU(),
                grid_eps=0.1,
                device=device
            )
        )
        
        # Remaining layers
        for _ in range(num_layers - 1):
            self.kan_layers.append(
                KANLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num=num_splines,
                    k=spline_order,
                    scale_base_mu=0.0,
                    scale_base_sigma=0.2,
                    scale_sp=1.0,
                    base_fun=nn.SiLU(),
                    grid_eps=0.1,
                    device=device
                )
            )
        
        # Output layer to predict next event
        self.output_layer = nn.Linear(hidden_dim, num_events)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Tensor of shape [batch_size, sequence_length] containing event indices
            return_attention: Whether to return attention weights for interpretability
        
        Returns:
            logits: Tensor of shape [batch_size, num_events] containing unnormalized
                   probabilities for the next event
            attention_weights: (Optional) Attention weights for interpretability
        """
        # Encode the sequence
        encoded = self.encoder(x)
        
        # Apply KAN layers
        kan_output = encoded
        attention_weights = None
        
        for i, layer in enumerate(self.kan_layers):
            kan_output, preacts, postacts, postspline = layer(kan_output)
            
            # Save attention weights from the last layer for interpretability
            if i == len(self.kan_layers) - 1 and return_attention:
                attention_weights = postacts
        
        # Output layer
        logits = self.output_layer(kan_output)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_next_event(self, sequence, idx_to_label, label_to_idx, temperature=1.0):
        """Predict the next event in the sequence with probability distribution"""
        with torch.no_grad():
            # Convert sequence to tensor of indices
            seq_indices = [label_to_idx[label] for label in sequence]
            
            # Pad if necessary
            if len(seq_indices) < self.sequence_length:
                seq_indices = [0] * (self.sequence_length - len(seq_indices)) + seq_indices
            else:
                seq_indices = seq_indices[-self.sequence_length:]
                
            seq_tensor = torch.LongTensor(seq_indices).unsqueeze(0).to(self.device)
            
            # Get logits
            logits = self(seq_tensor)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Convert to probabilities
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # Get the most likely next event and its probability
            next_event_idx = torch.argmax(probabilities, dim=-1).item()
            next_event_label = idx_to_label[next_event_idx]
            next_event_prob = probabilities[0, next_event_idx].item()
            
            # Get full probability distribution
            prob_distribution = {
                idx_to_label[i]: probabilities[0, i].item() 
                for i in range(len(idx_to_label))
                if i > 0  # Skip padding token
            }
            
            return next_event_label, next_event_prob, prob_distribution
    
    def generate_sequence(self, prefix_sequence, max_length, idx_to_label, label_to_idx, temperature=1.0):
        """Generate the rest of a sequence given a prefix"""
        with torch.no_grad():
            sequence = prefix_sequence.copy()
            probabilities = []
            
            # Generate events until we reach max_length
            while len(sequence) < max_length:
                # Get the next event prediction
                next_event, prob, prob_dist = self.predict_next_event(
                    sequence, idx_to_label, label_to_idx, temperature
                )
                
                # Add to our generated sequence
                sequence.append(next_event)
                probabilities.append((next_event, prob, prob_dist))
                
            return sequence, probabilities


class MultiScaleEventSequenceKAN(nn.Module):
    """
    Multi-scale KAN model for event sequence prediction
    Processes the sequence at multiple temporal scales for better long-term dependencies
    """
    def __init__(self, num_events, hidden_dim=64, embedding_dim=32, 
                 num_layers=2, sequence_length=10, 
                 num_splines=5, spline_order=3, device='cpu'):
        super(MultiScaleEventSequenceKAN, self).__init__()
        
        self.num_events = num_events
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.device = device
        
        # Embedding layer for event tokens
        self.embedding = nn.Embedding(num_events, embedding_dim)
        
        # Positional encoding
        position = torch.arange(0, sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                            -(np.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(sequence_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Multi-scale processing
        # 1. Local scale (individual events)
        self.local_kan = self._create_kan_block(embedding_dim, hidden_dim, num_layers, 
                                            num_splines, spline_order, device)
        
        # 2. Mid-range scale (subsequences)
        self.mid_kan = self._create_kan_block(embedding_dim, hidden_dim, num_layers, 
                                          num_splines, spline_order, device)
        
        # 3. Global scale (entire sequence)
        self.global_kan = self._create_kan_block(embedding_dim, hidden_dim, num_layers, 
                                             num_splines, spline_order, device)
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, num_events)
    
    def _create_kan_block(self, in_dim, hidden_dim, num_layers, num_splines, spline_order, device):
        """Create a block of KAN layers"""
        layers = nn.ModuleList()
        
        # First layer
        layers.append(
            KANLayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                num=num_splines,
                k=spline_order,
                base_fun=nn.SiLU(),
                device=device
            )
        )
        
        # Remaining layers
        for _ in range(num_layers - 1):
            layers.append(
                KANLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num=num_splines,
                    k=spline_order,
                    base_fun=nn.SiLU(),
                    device=device
                )
            )
            
        return layers
    
    def _process_kan_block(self, x, block):
        """Process through a KAN block"""
        for layer in block:
            x, _, _, _ = layer(x)
        return x
    
    def forward(self, x):
        # x shape: [batch_size, seq_length]
        batch_size = x.size(0)
        
        # Embed events
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Add positional encoding
        embedded = embedded + self.pe.unsqueeze(0)
        
        # 1. Local scale processing
        local_features = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        for i in range(self.sequence_length):
            # Process each position separately
            local_input = embedded[:, i, :]  # [batch_size, embedding_dim]
            local_out = self._process_kan_block(local_input, self.local_kan)
            local_features += local_out / self.sequence_length
        
        # 2. Mid-range scale processing
        mid_features = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        step = max(1, self.sequence_length // 3)  # divide sequence into 3 windows
        for i in range(0, self.sequence_length - step + 1, step):
            # Process each window
            window = embedded[:, i:i+step, :].mean(dim=1)  # [batch_size, embedding_dim]
            mid_out = self._process_kan_block(window, self.mid_kan)
            mid_features += mid_out / ((self.sequence_length // step) or 1)
        
        # 3. Global scale processing
        global_input = embedded.mean(dim=1)  # [batch_size, embedding_dim]
        global_features = self._process_kan_block(global_input, self.global_kan)
        
        # Combine features with learnable weights
        norm_weights = F.softmax(self.scale_weights, dim=0)
        combined = (
            norm_weights[0] * local_features + 
            norm_weights[1] * mid_features + 
            norm_weights[2] * global_features
        )
        
        # Output projection
        logits = self.output_layer(combined)
        
        return logits
    
    def predict_next_event(self, sequence, idx_to_label, label_to_idx, temperature=1.0):
        """Predict the next event in the sequence with probability distribution"""
        with torch.no_grad():
            # Convert sequence to tensor of indices
            seq_indices = [label_to_idx[label] for label in sequence]
            
            # Pad if necessary
            if len(seq_indices) < self.sequence_length:
                seq_indices = [0] * (self.sequence_length - len(seq_indices)) + seq_indices
            else:
                seq_indices = seq_indices[-self.sequence_length:]
                
            seq_tensor = torch.LongTensor(seq_indices).unsqueeze(0).to(self.device)
            
            # Get logits
            logits = self(seq_tensor)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Convert to probabilities
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # Get the most likely next event and its probability
            next_event_idx = torch.argmax(probabilities, dim=-1).item()
            next_event_label = idx_to_label[next_event_idx]
            next_event_prob = probabilities[0, next_event_idx].item()
            
            # Get full probability distribution
            prob_distribution = {
                idx_to_label[i]: probabilities[0, i].item() 
                for i in range(len(idx_to_label))
                if i > 0  # Skip padding token
            }
            
            return next_event_label, next_event_prob, prob_distribution
    
    def generate_sequence(self, prefix_sequence, max_length, idx_to_label, label_to_idx, temperature=1.0):
        """Generate the rest of a sequence given a prefix"""
        with torch.no_grad():
            sequence = prefix_sequence.copy()
            probabilities = []
            
            # Generate events until we reach max_length
            while len(sequence) < max_length:
                # Get the next event prediction
                next_event, prob, prob_dist = self.predict_next_event(
                    sequence, idx_to_label, label_to_idx, temperature
                )
                
                # Add to our generated sequence
                sequence.append(next_event)
                probabilities.append((next_event, prob, prob_dist))
                
            return sequence, probabilities


def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001):
    """Train the event sequence KAN model"""
    device = model.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_weights = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
    
    # Load the best model weights
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model, train_losses, val_losses
