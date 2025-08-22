import torch
import torch.nn as nn

class SimpleGameNN(nn.Module):
    def __init__(self):
        super(SimpleGameNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),        
            nn.Linear(11*11, 100), 
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Tanh()
  

        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, x):
        return self.model(x)
    

    