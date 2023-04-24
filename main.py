import Utilities as util
import cv2
import matplotlib.pyplot as plt
import kmeans as km
import deepKmeans as dk
import numpy as np
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models as tv
import agglomerative as agg

device = "cuda" if torch.cuda.is_available() else "cpu"

def kmeans_test():
    img = util.read_image('test.jpg')
    vectorized = util.image_to_vector(img)

    
    # run the kmeans algorithm
    kmeans = km.kmeans(dim = 3, k = 2, data = vectorized)
    kmeans.run()
    res = kmeans.process_data([0,255])
    centroids = kmeans.centroids
    
    res1 = np.zeros((vectorized.shape[0],3))
    # am array as long as the number of pixels in the image
    # if the pixel is in the first cluster, the value is as same as the first centroid, otherwise the value is as same as the second centroid
    for i in range(res.shape[0]):
        if res[i] == 0:
            res1[i] = centroids[0]
        else:
            res1[i] = centroids[1]
    # display the image in rgb space again
    # if the culster is 0, set the color same as the first centroid, otherwise set the color same as the second centroid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vectorized[:,0],vectorized[:,1],vectorized[:,2],c=res1/255)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    # set the color of the axis label
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    plt.show()
    
    
    # reshape the result
    res = res.reshape(img.shape[0],img.shape[1])
    # save the result  as a image
    cv2.imwrite('result.jpg',res)
    
def agglomerative_test():
    # random generate a data in dim 3(rgb) and 1000 points
    # data = np.random.randint(0,255,(100,3))
    # sample 100 points from the gaussian distribution in RGB space
    data = np.random.multivariate_normal([0,0,0], [[1,0,0],[0,1,0],[0,0,1]], 100)
    # normalize the data in the range of [0,255]
    data = (data - np.min(data))/(np.max(data) - np.min(data)) * 255
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],c=data/255)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    # set the color of the axis label
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    plt.show()
    
    # run the agglomerative algorithm, and set the number of clusters to 3
    agglomerative = agg.AgglomerativeClustering( k = 5, data = data)
    agglomerative.fit()
    # get the result
    res = agglomerative.get_result()
    # normalize the result in the range of [0,255]
    res = (res - np.min(res))/(np.max(res) - np.min(res)) * 255
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],c=res/255)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    # set the color of the axis label
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    plt.show()
    
def draw_Gaussian():
    # draw a 2D gaussian distribution in RGB space, using contiuous color 
    # to represent the probability of the point in the distribution
    # value is range from 0 to 255
    # the coordinate is in RGB space, and for each point in the distribution, its color is the same as the point
    # the color of the axis label is red, green and blue
    x = np.linspace(0,255,100)
    y = np.linspace(0,255,100)
    X,Y = np.meshgrid(x,y)
    Z = np.exp(-((X-127.5)**2 + (Y-127.5)**2)/10000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z,cmap='jet')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    # set the color of the axis label
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')
    ax.zaxis.label.set_color('blue')
    plt.show()
    
def Eval_Kmeans():
    # for each images in the floder imgs, run the kmeans algorithm and save the result
    # the result is saved in the folder results
    avg_acc = 0
    for i in range(1,16):
        # print("processing image "+str(i))
        img = util.read_image('imgs/'+str(i)+'.jpg')
        # convert the image to grayscale
        vectorized = util.image_to_vector(img)
        # run the kmeans algorithm
        kmeans = km.kmeans(dim = 3, k = 2, data = vectorized)
        kmeans.run()
        res = kmeans.process_data([0,255])
        
        # read the ground truth from the folder gt
        gt = util.read_image('gt/'+str(i)+'.png')
        # calculate the accuracy using Mean Square Error
        acc = util.accuracy(res,gt)
        avg_acc = avg_acc + acc
    avg_acc = avg_acc/16
    print(avg_acc)


# if use the feature to cluster algorithm, set the deep to True
def deep_kmeans_test(scale=4):
    # tarverse the folder data
    data = []
    vgg = util.VGG().to(device)
    for i in range(0,20):
        # print("processing image "+str(i+1))
        if scale == 0:
            img = util.get_img_pixel('data/'+str(i+1)+'.jpg')
        else:
            img = util.get_feature('data/'+str(i+1)+'.jpg',vgg,scale=scale)
        
        # add the feature to the data
        data.append(img)
    # apply the deepkmeans algorithm to data
    deepkmeans = dk.deepKmeans(k = 2, data = data)
    # run the algorithm
    deepkmeans.train()
    # get the cluster result
    res = deepkmeans.get_cluster()
    err = 0
    for i in range(0,20):
        if res[i] != i // 10:
            err = err + 1
    return err
    
if __name__ == '__main__':
    # Eval_Kmeans()
    iter_num = 1
    image_num = 20
    results = []
    for scale in range(0,6):
        print("scale = "+str(scale))
        err = 0
        for i in range(0,iter_num):
            print("iter = "+str(i))
            err = err + deep_kmeans_test(scale=scale)
            print("err = "+str(deep_kmeans_test(scale=scale))+"  acc = "+str(1-err/(i+1)/image_num))
        err = err / iter_num / image_num
        results.append(err)
    print(results)
    # kmeans_test()
    # agglomerative_test()
    # draw_Gaussian()
