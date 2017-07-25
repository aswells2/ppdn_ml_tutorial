import numpy as np
from sklearn.cluster import MeanShift #as ms
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1,1],[5,5],[3,10]]

X, _ = make_blobs(n_samples = 500, centers = centers, cluster_std = 1)

Z, z = make_blobs(n_samples = 10, centers = centers, cluster_std = .8)


for i in z:
    1==1
    2==0
    0==2

for i in range(len(Z)):
    print(Z[i][0],Z[i][1],z[i])


plt.scatter(X[:,0],X[:,1])
plt.scatter(Z[:,0],Z[:,1])

plt.show()


ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labells=ms.predict(Z) #notice the double L. Labells are provided for values of Z and then ploted in line 44 below
print(ms.labels_)

print(cluster_centers)

n_clusters_ = len(np.unique(labels))

print("Number of estimated clusters:", n_clusters_)

colors = 10*['r.','g.','b.','c.','k.','y.']
colours = 10*['r*','g*','b*','c*','k*','y*']#changes shape of prediction plots (line 44).


#print(colors)
##print(labels)
##print(len(labells))




for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 10)
for k in range(len(Z)):
    plt.plot(Z[k][0],Z[k][1], colours[labells[k]], markersize = 20)
for k in range(len(labells)):
    print(Z[k][0],Z[k][1],labells[k])

plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
            marker="x", s=250, linewidths = 5, zorder=10)
plt.show()
