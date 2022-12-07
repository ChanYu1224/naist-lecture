import torch

# initialization of inputs
H = torch.tensor([
    [0.5, -1.2],
    [-0.2, 0.8],
])

x = torch.tensor([
    [0.7],
    [-0.3],
])

d = torch.tensor([
    [0.1],
    [0.2],
])

U = torch.tensor([
    [0.3, -0.5, 0.2, -0.1],
    [-0.1, 0.3, 0.4, 0.1],
    [0.1, 0.5, 0.2, -0.2],
])

b = torch.tensor([
    [0.3],
    [-0.2],
    [0.1],
])

label = torch.tensor([
    [0.2],
    [0.8],
    [0.0],
])

# computation
f1 = torch.matmul(H, x)
f2 = f1 + d
f3 = torch.tanh(f2)
f4 = torch.concat([f3, x], dim=0)
f5 = torch.matmul(U, f4)
f6 = f5 + b

print("output:", f6)
print("label :", label)

x = f6.squeeze()
y = label.squeeze()
loss = torch.nn.functional.cross_entropy(x, y)

print("loss  :", loss)