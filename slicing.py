import torch
import numpy as np

x = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])
y = np.array([[1,2,3],[4,5,6],[7,8,9]])
w = x.numpy() #conversion to numpy
z = torch.from_numpy(y)
print(z)

# print("print the first of each row ",x[:,0])
# print("print each element in the first row",y[0,:])
print(x.shape)
print("print each element in the first row",x[:,:,:])

#numpy can handle only cpu tensor

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) #creating a tensor on the gpu
    y = torch.ones(5)
    y = y.to(device)
    z = x*y
    z = z.to("cpu") #moving the tensor on the cpu
