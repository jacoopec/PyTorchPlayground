import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

# n_samples, n_features = iris.data.shape
# print(iris.keys())
# print('Number of samples:', n_samples) #150
# print('Number of features:', n_features) #4 features
# print(iris.keys())
# print(iris["target_names"])
# # the sepal length, sepal width, petal length and petal width of the first sample (first flower)

# print(iris.data[0]) #first element
# print(iris.data[0].target) #first's element target 

fig, ax = plt.subplots()

x_index = 0
y_index = 4

colors = ['blue', 'red', 'green']

for label, color in zip(range(len(iris.target_names)), colors):
    ax.scatter(iris.data[iris.target==label, x_index], 
                iris.data[iris.target==label, y_index],
                label=iris.target_names[label],
                c=color)

ax.set_xlabel(iris.feature_names[x_index])
ax.set_ylabel(iris.feature_names[y_index])
ax.legend(loc='upper left')
plt.show()