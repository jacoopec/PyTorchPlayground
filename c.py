import torch
import numpy as np

a = torch.ones(4, dtype=torch.int)
a2 = torch.tensor([3.4, 2.3])
a3 = torch.tensor([23.4,23.4])
sum2 = a2 + a3
sum = torch.add(a2,a3)
sum3 = a2.add_(a3) 
mul = torch.mul(a2, a3)
mul2 = a2.mul(a3)
# a.add_(1) adding to all a element a 1

empt = torch.empty(3,2,2)
print(empt)
rand = torch.rand(2,3)
print(a)
b = a.numpy()
# c = torch.from_numpy(a, dtype="int32")
# print(type(b))

