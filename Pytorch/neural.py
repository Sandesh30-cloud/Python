import torch
import torch.nn as nn
# One neuron
model = nn.Linear(2, 1)
# Input
x = torch.tensor([[2.0, 3.0]])
# Output
y = model(x)
print(y)
#  uv run neural.py 
# tensor([[0.5920]], grad_fn=<AddmmBackward0>)




#sequential model
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),   # Input → Hidden
    nn.ReLU(),          # Activation
    nn.Linear(4, 1)     # Hidden → Output
)

x = torch.tensor([[1.0, 2.0]])
output = model(x)

print(output)




# 3. Custom Neural Network (Recommended for Interviews)
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNetwork()

x = torch.tensor([[1.0, 2.0]])
print(model(x))







# 4. Tiny Training Example
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model
model = nn.Linear(1, 1)

# Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train
for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[5.0]])))
# tensor([[9.8275]], grad_fn=<AddmmBackward0>)

'''
The model learns the relationship: y = 2x
So the prediction for 5 will be close to: 10
'''