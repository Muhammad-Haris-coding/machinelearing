import random
import numpy as np
# deside cluster
#select random centroids


class KMean:
    def __init__(self, n_clusters=2 , max_iter=300):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.centroids=None

    def fit_pridect(self,X):
        random_ind=random.sample(range(0,X.shape[0]),self.n_clusters)
        centroids=X[random_ind]
        
        for i in range (self.max_iter):
            #assign clusters
            cluster_gorup=self.assign_clusters(X,centroids)
            #move centroids
            old_centroids=centroids.copy()
            centroids=self.move_centroids(X,cluster_gorup)

            #check finish
            if (old_centroids==centroids).all():
                break
        self.centroids=centroids
        return cluster_gorup

    def assign_clusters(self,x,centroids):
        cluster_group=[]
        distences=[]
        for row in x:
            for centroid in centroids:
                distences.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distances=min(distences)
            min_indx=distences.index(min_distances)
            cluster_group.append(min_indx)
            distences.clear()
        return np.array(cluster_group)

    def move_centroids(self,x,cluster_group):
        new_centroids=[]
        cluster_type=np.unique(cluster_group)
        for type in cluster_type:
            new_centroids.append(x[cluster_group==type].mean(axis=0))
        return np.array(new_centroids)