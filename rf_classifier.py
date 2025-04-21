import torch
import torch.nn as nn

class RFClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_classes=33, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Third hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),

            # Output layer (no BN needed)
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)