from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

wines = load_wine()

print(wines.keys())
# print(wines.feature_names)
# print(wines.DESCR)
print(wines.data[0])

fig = plt.figure(figsize=(6, 6)) 
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(wines.DESCR[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(wines.target[i]))
