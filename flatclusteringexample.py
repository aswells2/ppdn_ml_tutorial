import numpy as np #to call up numpy; used for arrays
import matplotlib.pyplot as plt 
from matplotlib import style #the next two lines are for pretty graphs (not needed)
style.use("ggplot")
from sklearn.cluster import KMeans #scikitlearn verbiage for running KMeans ML

x = [1,5,1.5,8,1,9]
y = [2,8,1.8,8,0.6,11]
'''
establishing array information. This small 6 coordinate data set will be clustered

'''
plt.scatter(x,y)

plt.show

#for visualizing the data "plt." is a matplotlib command

X = np.array([[1, 2],
              [5,8],
              [1.5,1.8],
              [8,8],
              [1,.6],
              [9,11]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
'''
numpy is being called in order to form an array(python has no native array functionality).
This array will be clustered
using n_clusters command. "=2" indicates we are separting them into two groups
Algorithm determines which coordinates fit into which clusters

'''

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)
'''
showing the center of the clusters, printing and labeling them as defined below

'''
colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    print("coordinate:", X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

'''
Basically, for each coordinate (len(X)), print and label the "ith" coordinate on the plot
'''

plt.scatter(centroids[:,0],centroids[:,1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()



