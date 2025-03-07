import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class KANLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KANLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        self.psi = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        # Apply phi to each input dimension
        phi_outputs = [self.phi(x[:, i:i+1]) for i in range(x.size(1))]
        phi_outputs = torch.stack(phi_outputs, dim=1)
        
        # Apply psi to each pair of phi outputs
        psi_outputs = []
        for i in range(x.size(1)):
            for j in range(i+1, x.size(1)):
                psi_outputs.append(self.psi(torch.cat((phi_outputs[:, i], phi_outputs[:, j]), dim=1)))
        psi_outputs = torch.stack(psi_outputs, dim=1)
        
        # Sum the psi outputs
        return torch.sum(psi_outputs, dim=1)

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList([KANLayer(input_dim, hidden_dim, output_dim) for _ in range(num_layers)])
        self.final_layer = nn.Linear(output_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class AttentionKANLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionKANLayer, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        self.psi = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        self.attention = nn.Linear(output_dim, 1)

    def forward(self, x):
        # Apply phi to each input dimension
        phi_outputs = [self.phi(x[:, i:i+1]) for i in range(x.size(1))]
        phi_outputs = torch.stack(phi_outputs, dim=1)
        
        # Apply attention to phi outputs
        attention_scores = self.attention(phi_outputs)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_phi_outputs = phi_outputs * attention_weights
        
        # Apply psi to each pair of weighted phi outputs
        psi_outputs = []
        for i in range(x.size(1)):
            for j in range(i+1, x.size(1)):
                psi_outputs.append(self.psi(torch.cat((weighted_phi_outputs[:, i], weighted_phi_outputs[:, j]), dim=1)))
        psi_outputs = torch.stack(psi_outputs, dim=1)
        
        # Sum the psi outputs
        return torch.sum(psi_outputs, dim=1)

class AttentionKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(AttentionKAN, self).__init__()
        self.layers = nn.ModuleList([AttentionKANLayer(input_dim, hidden_dim, output_dim) for _ in range(num_layers)])
        self.final_layer = nn.Linear(output_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

# Example usage
input_dim = 3
hidden_dim = 10
output_dim = 5
num_layers = 2

# Create models
model_base = KAN(input_dim, hidden_dim, output_dim, num_layers)
model_attention = AttentionKAN(input_dim, hidden_dim, output_dim, num_layers)

# Generate toy dataset
def generate_toy_dataset(num_samples, input_dim):
    x = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, 1)  # Random labels
    return x, y

# Generate dataset
num_samples = 100
x, y = generate_toy_dataset(num_samples, input_dim)

# Define loss function and optimizers
criterion = nn.MSELoss()
optimizer_base = optim.Adam(model_base.parameters(), lr=0.01)
optimizer_attention = optim.Adam(model_attention.parameters(), lr=0.01)

# Training loop for base model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer_base.zero_grad()
    output_base = model_base(x)
    loss_base = criterion(output_base, y)
    loss_base.backward()
    optimizer_base.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Base Model Loss: {loss_base.item()}")

# Training loop for attention model
for epoch in range(num_epochs):
    optimizer_attention.zero_grad()
    output_attention = model_attention(x)
    loss_attention = criterion(output_attention, y)
    loss_attention.backward()
    optimizer_attention.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Attention Model Loss: {loss_attention.item()}")

# Evaluate the models
with torch.no_grad():
    output_base = model_base(x)
    output_attention = model_attention(x)
    loss_base = criterion(output_base, y)
    loss_attention = criterion(output_attention, y)
    print(f"Final Base Model Loss: {loss_base.item()}")
    print(f"Final Attention Model Loss: {loss_attention.item()}")
