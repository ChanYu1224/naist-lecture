from turtle import forward
import torch
from torch import nn

import numpy as np

# hyperparameter
learning_rate = 1.00

# initial value
student_id = '2211224'
b = list(map(int, student_id[3:7]))

# set weight
w = np.zeros((3,7))
w[1][1] = (b[2]+2)/10
w[1][2] = -(b[3]+2)/10
w[1][3] = (b[1]+2)/10
w[1][4] = (b[1]+2)/10
w[1][5] = (b[2]+2)/10
w[1][6] = -(b[3]+2)/10
w[2][1] = -(b[1]+2)/10
w[2][2] = -(b[2]+2)/10
w[2][3] = -(b[3]+2)/10


class MLP(nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(2,2),
            nn.Tanh(),
            nn.Linear(2,1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.linear_stack(x)
    


def train(model, optimizer, loss_function):
    X = torch.tensor([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ], dtype=torch.float)
    y = torch.tensor([
        [1], 
        [-1], 
        [-1],
        [1],
    ], dtype=torch.float)

    for t, x in enumerate(X):
        predicted = model(x)
        loss = loss_function(predicted, y[t])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('--- step', str(t+1), '---')
        print(model.state_dict())
        print('predicted :', predicted.item())
        print('true y    :', y[t].item())
        print('loss      :', loss.item())


class SquareError(nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, predicted, y):
        return torch.pow(torch.sub(predicted, y), 2)/2


def main():
    model = MLP()

    # set weights
    weights = model.state_dict()
    weights['linear_stack.0.weight'] = torch.tensor([
    [w[1][1], w[1][3]],
    [w[1][2], w[1][4]],
    ])
    weights['linear_stack.0.bias'] = torch.tensor([w[1][5], w[1][6]])
    weights['linear_stack.2.weight'] = torch.tensor([
        [w[2][1], w[2][2]]
    ])
    weights['linear_stack.2.bias'] = torch.tensor([w[2][3]])
    
    # write weights
    model.load_state_dict(weights)

    # select optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = SquareError()

    # display weight
    print(model.state_dict())

    # optimization
    train(model=model, optimizer=optimizer, loss_function=loss_function)


if __name__=='__main__':
    main()
