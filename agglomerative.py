'''
@description: Agglomerative Clustering
@author: chtholly
@date: 2023-4-22
'''
import numpy as np

class AgglomerativeClustering:
    def __init__(self, data, k):
        '''
        @description: initialize the AgglomerativeClustering class
        @param {numpy.ndarray} data: the data to be clustered, which is a n*dim matrix
        @param {int} k: the number of clusters
        '''
        # the data to be clustered, which is a n*dim matrix
        # at begin the number of clusters is equal to the number of points
        # cluster number is reduced by 1 in each iteration, and the elements in the data matrix are reduced by 1
        self.data = data
        # number of clusters, each point is a cluster at the beginning
        self.n = data.shape[0]
        # the number of clusters
        self.k = k
        # initialize the cluster, each point is assigned to a cluster, value range: 0~k-1
        self.cluster = np.zeros((self.n, 1))
        self.cluster_size = np.zeros((self.n, 1))
        # initialize the distance matrix, the distance between two points
        self.distance = np.zeros((self.n, self.n))
        self.distance.fill(float('inf'))
        # initialize the cluster distance matrix, the distance between two clusters
        self.cluster_distance = np.zeros((self.n, self.n))
        self.cluster_distance.fill(float('inf'))
        # set the diagonal of the distance matrix to be inf, in order to avoid the point to be clustered to itself
        self.cluster_distance = self.cluster_distance + np.diag(np.ones(self.n) * float('inf'))
        
    def cal_distance(self, point1, point2):
        '''
        @description: calculate the distance between two points in n dimensions
        @param {numpy.ndarray} point1: the first point
        @param {numpy.ndarray} point2: the second point
        @return {float} distance: the distance between the two points
        '''
        distance = np.sqrt(np.sum(np.square(point1 - point2)))
        return distance

    def update_distance(self):
        '''
        @description: update the distance matrix
        '''
        # for each point, calculate the distance between it and the other points
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.distance[i, j] = self.cal_distance(self.data[i, :], self.data[j, :])
                self.distance[j, i] = self.distance[i, j]

    def update_cluster_distance(self):
        '''
        @description: update the cluster distance matrix
        '''
        # for each cluster, calculate the distance between it and the other clusters
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # use self.cluster_size[i] * self.data[i, :] / self.cluster_size[i] instead of self.data[i, :] to avoid the cluster size is 0
                self.cluster_distance[i, j] = self.cal_distance(self.cluster_size[i] * self.data[i, :] / self.cluster_size[i],
                                                                self.cluster_size[j] * self.data[j, :] / self.cluster_size[j])
                self.cluster_distance[j, i] = self.cluster_distance[i, j]

    def find_closest_cluster(self):
        '''
        @description: find the closest cluster
        @return {int} index1: the index of the first cluster
        @return {int} index2: the index of the second cluster
        '''
        min_distance = float('inf')
        index1 = -1
        index2 = -1
        # find the two clusters with the minimum distance, and return their indices
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.cluster_distance[i, j] < min_distance:
                    min_distance = self.cluster_distance[i, j]
                    index1 = i
                    index2 = j
        return index1, index2

    def update_cluster(self, index1, index2):
        '''
        @description: update the cluster
        @param {int} index1: the index of the first cluster
        @param {int} index2: the index of the second cluster
        '''
        # update the cluster, the points in the second cluster are assigned to the first cluster
        self.cluster[np.where(self.cluster == index2)[0]] = index1
        # update the cluster size
        self.cluster_size[index1] = self.cluster_size[index1] + self.cluster_size[index2]
        self.cluster_size[index2] = 0

    def update_data(self, index1, index2):
        '''
        @description: update the data
        @param {int} index1: the index of the first cluster
        @param {int} index2: the index of the second cluster
        '''
        # update the data, the points in the second cluster are assigned to the first cluster
        self.data[index1, :] = self.cluster_size[index1] * self.data[index1, :] / self.cluster_size[index1] + self.cluster_size[index2] * self.data[index2, :] / self.cluster_size[index2]
        self.data[index2, :] = 0
        

    def fit(self):
        '''
        @description: fit the data
        '''
        # initialize the cluster size
        self.cluster_size = np.ones((self.n, 1))
        self.update_distance()
        self.update_cluster_distance()
        # merge the clusters until the number of clusters is equal to k
        for i in range(self.n - self.k):
            # find the two clusters with the minimum distance
            index1, index2 = self.find_closest_cluster()
            # update the cluster, the points in the second cluster are assigned to the first cluster
            self.update_cluster(index1, index2)
            self.update_data(index1, index2)
            # update the distance matrix and the cluster distance matrix
            self.update_cluster_distance()
            
    def get_result(self):
        '''
        @description: get the clustering result
        @return {numpy.ndarray} cluster: the clustering result
        '''
        return self.data
            

