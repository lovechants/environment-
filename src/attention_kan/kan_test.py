import torch
import torch.nn as nn
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

# Example usage
input_dim = 3
hidden_dim = 10
output_dim = 5
num_layers = 2

model = KAN(input_dim, hidden_dim, output_dim, num_layers)

# Example input
x = torch.randn(10, input_dim)

# Forward pass
output = model(x)
print(output)
