'''
@description: implement the kmeans algorithm, using the deep feature of the image as the data
@author: chtholly
@date: 2023-4-22
'''

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models as tv
import math

class deepKmeans:
    def __init__(self,data,k,max_iter = 200,device = 'cuda'):
        '''
        @description: initialize the kmeans class
        @param {numpy.ndarray} data: a list, each element is a torch.Tensor, represent the features from the last layer of the VGG
        @param {int} k: the number of clusters
        @param {int} max_iter: the maximum number of iterations
        ''' 
        self.max_iter = max_iter
        self.data = data
        self.k = k
        # randomly initialize the centroids,each centroid is a torch.Tensor, with the same shape as the feature in self.data
        self.centroids = []
        # print the type of the data[0]
        # first centorids is the mean of data[0]-data[9]
        # second centorids is the mean of data[10]-data[19]
        # ...
        # last centorids is the mean of data[10*(k-1)]-data[10*k-1]
        for i in range(k):
            self.centroids.append(torch.mean(torch.stack(self.data[10*i:10*(i+1)]),dim = 0))
        # initialize the cluster, each point is assigned to a cluster, value range: 0~k-1, cluster is a int numpy.ndarray
        self.cluster = np.ones((len(data),1))
        print(self.cluster)
        
        # number of points in each cluster
        self.cluster_size = np.zeros((k,1))
        
        
        
    def cal_distance(self,point1,point2):
        '''
        @description: calculate the distance between two points in feature space, each point is a torch.Tensor
        @param {torch.Tensor} point1: the first point in feature space
        @param {torch.Tensor} point2: the second point in feature space
        @return {float} distance: the L_2 distance between the two points
        '''
        distance = torch.sqrt(torch.sum(torch.square(point1-point2)))
        return distance
    
    def find_closest_centroid(self,point):
        '''
        @description: find the closest centroid to a point
        @param {torch.Tensor} point: the point to be clustered
        @return {int} index: the index of the closest centroid
        '''
        # min_distance is also a torch.Tensor, with value float('inf') at the beginning
        min_distance = torch.tensor(float('inf'))
        index = -1
        for i in range(self.k):
            distance = self.cal_distance(point,self.centroids[i])
            if distance < min_distance:
                min_distance = distance
                index = i
        return int(index)
    
    def update_centroids(self):
        '''
        @description: update the centroids
        '''
        # for each centroid, calculate the mean of the points in the cluster, and update the centroid
        for i in range(self.k):
            points = []
            for j in range(len(self.data)):
                if self.cluster[j] == i:
                    points.append(self.data[j])
            if len(points) == 0:
                continue
            points = torch.stack(points)
            self.centroids[i] = torch.mean(points,dim = 0)

    def update_cluster(self):
        '''
        @description: update the cluster
        '''
        # reset the cluster_size
        self.cluster_size = np.zeros((self.k,1))
        
        for i in range(len(self.data)):
            self.cluster[i] = self.find_closest_centroid(self.data[i])
            self.cluster_size[int(self.cluster[i])] += 1
            
    def is_converged(self):
        '''
        @description: check whether the algorithm is converged
        @return {bool} is_converged: whether the algorithm is converged
        '''
        converged = True
        for i in range(len(self.data)):
            if self.find_closest_centroid(self.data[i]) != self.cluster[i]:
                converged = False
                break
        return converged
    
    def train(self):
        '''
        @description: train the model
        '''
        iter = 0
        for i in range(self.max_iter):
            self.update_centroids()
            self.update_cluster()
            iter += 1
            print('iter: ',iter)
            print('cluster size: ',self.cluster_size)
            if self.is_converged():
                break
            
    def get_cluster(self):
        '''
        @description: get the cluster
        @return {numpy.ndarray} cluster: the cluster
        '''
        return self.cluster
        

