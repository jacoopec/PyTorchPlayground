import numpy as np

from sklearn.model_selection import train_test_split

array = [0,1,2,3,4,5,6,7,8,9]
labels = [0,1,0,1,1,1,0,0,0,1]

print(array[-4:])
print(np.random.permutation(array))


res = train_test_split(array, labels, train_size=0.8, test_size=0.2, random_state=42)

train_data, test_data, train_labels, test_labels = res 

print(train_data)
print(test_data)

