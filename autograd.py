#calculating gradients
import torch
#autograd is used to calculate gradients
x = torch.randn(3, requires_grad=True)  #True is necessary to call backward
#whenever we will do operations with the tensor x pytorch will create the computational graph
#we want to calculate the gradient of some function with respect to x
y = x + 2 #creating computational graph
print(x)
print(y)
z = y*y*2 #the gradient function is mul_backward
z = z.mean() #the gradient function is meanbackward
v = torch.tensor([0.1,2.0,0.3], dtype=torch.float32)
v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward(v) #dz/dx caltulating the gradient

print(x.grad) #in this tensor we have the gradients

