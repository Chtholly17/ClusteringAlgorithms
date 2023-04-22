'''
@description: implement the kmeans algorithm in n dimensions
@author: chtholly
@date: 2023-4-22
'''

import numpy as np

class kmeans:
    def __init__(self,dim,data,k,max_iter = 100):
        '''
        @description: initialize the kmeans class
        @param {int} dim: the dimension of the data
        @param {numpy.ndarray} data: the data to be clustered,which is a n*dim matrix
        @param {int} k: the number of clusters
        ''' 
        self.max_iter = max_iter
        self.dim = dim
        self.data = data
        self.k = k
        # randomly initialize the centroids
        self.centroids = np.random.rand(k,dim)
        # initialize the cluster, each point is assigned to a cluster, value range: 0~k-1
        self.cluster = np.zeros((data.shape[0],1))
        # number of points in each cluster
        self.cluster_size = np.zeros((k,1))
    
    def cal_distance(self,point1,point2):
        '''
        @description: calculate the distance between two points in n dimensions
        @param {numpy.ndarray} point1: the first point
        @param {numpy.ndarray} point2: the second point
        @return {float} distance: the distance between the two points
        '''
        distance = np.sqrt(np.sum(np.square(point1-point2)))
        return distance
    
    def find_closest_centroid(self,point):
        '''
        @description: find the closest centroid to a point
        @param {numpy.ndarray} point: the point to be clustered
        @return {int} index: the index of the closest centroid
        '''
        min_distance = float('inf')
        index = -1
        for i in range(self.k):
            distance = self.cal_distance(point,self.centroids[i,:])
            if distance < min_distance:
                min_distance = distance
                index = i
        return index
    
    def update_centroids(self):
        '''
        @description: update the centroids
        '''
        # for each centroid, calculate the mean of the points in the cluster, and update the centroid
        for i in range(self.k):
            points = self.data[np.where(self.cluster == i)[0]]
            self.centroids[i,:] = np.mean(points,axis=0)
            
    def update_cluster(self):
        '''
        @description: update the cluster
        '''
        # for each point, find the closest centroid and update the cluster
        for i in range(self.data.shape[0]):
            self.cluster[i] = self.find_closest_centroid(self.data[i,:])
            self.cluster_size[int(self.cluster[i])] += 1
            
    def is_converged(self):
        '''
        @description: check if the cluster is converged
        @return {bool} converged: True if the cluster is converged, False otherwise
        '''
        converged = True
        for i in range(self.data.shape[0]):
            if self.cluster[i] != self.find_closest_centroid(self.data[i,:]):
                converged = False
                break
        return converged
            
    def run(self):
        '''
        @description: run the kmeans algorithm
        '''
        iter = 0
        # run the kmeans algorithm
        while True:
            # update the cluster
            self.update_cluster()
            # update the centroids
            self.update_centroids()
            # check if the cluster is converged
            
            # status display
            iter += 1
            print('iteration: ',iter)
            # display the number of points in each cluster
            print("cluster size: " ,self.cluster_size)
            # reset the cluster size
            self.cluster_size = np.zeros((self.k,1))
            
            # if the algorithm is converged or the maximum number of iterations is reached, stop the algorithm
            if self.is_converged() or iter > self.max_iter:
                break
        return self.cluster
    
    def process_data(self,values):
        res = np.zeros((self.cluster.shape[0],1))
        # traverse the data
        for i in range(self.cluster.shape[0]):
            res[i] = values[int(self.cluster[i])]     
        return res