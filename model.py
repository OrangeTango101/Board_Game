import torch
import torch.nn as nn

class SimpleGameNN(nn.Module):
    def __init__(self):
        super(SimpleGameNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),        
            nn.Linear((11*11)+2, 200), 
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)
    

    