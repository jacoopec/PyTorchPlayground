import torch 
import torch.nn as nn
import numpy as np
import sklearn import datasets
import matplotlib.pyplot as plt

#prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=2, noise = 20, random_state=1)
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples , n_features = x.shape
#model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
#define loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)
#trining loop
num_epochs = 100
for epoch in range(num_epochs):
    y_predicted = model(x)
    loss = criterion(y_predicted, y)
    
    loss.backward()
    
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch+1) %10 == 0:
        print(f'epoch: {epoch+1}, loss ={loss.item():4f}')

predicted = model(x).detach()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()




