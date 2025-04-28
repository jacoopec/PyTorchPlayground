import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
labels = iris.target
n_test_samples = 140

print(data[0])
print(len(data))
indices = np.random.permutation(len(data))
learn_data = data[indices[:-n_test_samples]]
print(learn_data)