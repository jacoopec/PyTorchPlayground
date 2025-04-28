#1 design model (input, output, size, forward pass)
#2 Construct the loss and optimizer
#3 do the training loop
    #forward pass: compute prediction
    #backward pass: gradients
    #update our weights
    #iterate a couple of time
    
import torch 
import torch.nn as nn



import torch

x = torch.tensor([[1],[2],[3],[4]], dtype= torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype= torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = 
print('n_samples %s, n_features %s'% n_samples, n_features)

input_size = n_features
output_size = n_features
# model = nn.Linear() using the class instead of this

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)


# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#------------------------------------
#replacing model prediction
# def forward(x):
#     return w*x
#------------------------------------

#------------------------------------
#We don't want to define the loss manually
#loss = MSE
# def loss(y, y_predicted):
#     return((y_predicted-y)**2).mean()


print(f'Prediction before training: f(5)={forward(5):.3f}')


#Training
learning_rate= 0.01
n_iters = 10

#new loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(x)
    #loss
    l = loss(y, y_pred)
    #gadients = backward pass
    l.backward() #dl/dw
 
#we don't have to compute the weights manually anymore   
    #update weights
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()
    
    # w.grad.zero()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0]:.3f},loss = {l:.8f}')
        
print(f'Prediction after the training: f(5)={forward(5):.3f}')

