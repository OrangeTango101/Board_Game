import torch
import torch.nn as nn

class SimpleGameNN(nn.Module):
    def __init__(self):
        super(SimpleGameNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),        
            nn.Linear(11*11, 30), 
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 1), 
        )
    def forward(self, x):
        return self.model(x)
    

    