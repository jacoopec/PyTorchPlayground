import torch
import numpy as np

x = torch.empty(3, dtype= torch.double)
x2 = torch.empty(3, dtype= torch.double)
zs = torch.zeros(3, dtype=torch.int)
rd = torch.rand(2,2)
t = torch.tensor([2.3, 0.2])
add = torch.add(x, x2)
x2.add_(x)
x3 = x - x2

print(rd)
print(x[1,1].item())