from sklearn.datasets import make_blobs
from kmean import KMean
import matplotlib.pyplot as plt

centroids=[(2,2), (8,3), (3,6)]
cluster_std=[1,1,1]

x,y =make_blobs(n_samples=100 , cluster_std=cluster_std , centers=centroids , n_features=2 , random_state=20)

km=KMean(n_clusters=3)
y_mean=km.fit_pridect(x)

plt.scatter(x[y_mean==0,0],x[y_mean==0,1], c='red')
plt.scatter(x[y_mean==1,0],x[y_mean==1,1], c='blue')
plt.scatter(x[y_mean==2,0],x[y_mean==2,1], c='green')
plt.show()