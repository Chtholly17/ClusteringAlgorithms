'''
@description: This file contains all the utility functions used in the project
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

device = "cuda" if torch.cuda.is_available() else "cpu"

def read_image(image_path):
    '''
    @description: Read the image from the given path,convert it to RGB and return it
    @param {string} image_path: The path of the image
    @return {numpy.ndarray} image: The image read from the given path
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_to_vector(image):
    '''
    @description: read a image in cv2 format and covert it to a vector using numpy
    @param {img} image: The image to be converted
    @return {numpy.ndarray} image: The image converted to a vector
    '''
    return image.reshape(-1, 3)

def accuracy(y_true, y_pred):
    '''
    @description: Calculate the Mean Square Error of the given prediction
    @param {numpy.ndarray} y_true: The true labels of the data
    @param {numpy.ndarray} y_pred: The predicted labels of the data
    @return {float} accuracy: The MSE of the prediction
    '''
    # Calculate the Mean Square Error of the given prediction
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, l2_filter=0):
        super(VGG, self).__init__()
        vgg_pretrained_features = tv.vgg19(pretrained=pretrained).features
            
        # print(vgg_pretrained_features)
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        
        # vgg16
        # for x in range(0,4):
        #     self.stage1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(4, 9):
        #     self.stage2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(9, 16):
        #     self.stage3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(16, 23):
        #     self.stage4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(23, 30):
        #     self.stage5.add_module(str(x), vgg_pretrained_features[x])

        # vgg19
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])                

        # for x in range(0,5):
        #     self.stage1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(5, 10):
        #     self.stage2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(10, 19):
        #     self.stage3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(19, 28):
        #     self.stage4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(28, 37):
        #     self.stage5.add_module(str(x), vgg_pretrained_features[x])  
                                            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,128,256,512,512]
        self.window_size = 3
        self.windows = self.create_window(self.window_size, self.window_size/3, 1)

    def gaussian(self,window_size, sigma, center = None):
        if center==None:
            center = window_size//2
        gauss = torch.Tensor([math.exp(-(x - center)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self,window_size, window_sigma, channel):
        _1D_window = self.gaussian(window_size, window_sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        # window = torch.ones_like(window)
        # window = window/window.sum(dim=[2,3],keepdim=True)
        return nn.Parameter(window,requires_grad=False)
  
    def get_features(self, x,scale=5):
        # normalize the data
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        # get the features of each layer
        # outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]#
        # output the feature of layer according to the scale   
        if scale == 1:
            outs = [h_relu1_2]
        elif scale == 2:
            outs = [h_relu2_2]
        elif scale == 3:
            outs = [h_relu3_3]
        elif scale == 4:
            outs = [h_relu4_3]
        elif scale == 5:
            outs = [h_relu5_3]
        # for i in range(0,len(outs)):
        #     outs[i] = F.normalize(outs[i])
            # outs[i] = outs[i] / (1e-12+torch.amax(outs[i],dim=[2,3], keepdim=True))
        return outs

    def construct_gau_pyramid(self, feats):
        f_pyr = [[] for i in range(len(feats))] 
        for i in range(len(feats)):
            f = feats[i]
            f_pyr[i].append(f)
            for j in range(i, len(feats)-1):
                win = self.windows.expand(f.shape[1], -1, -1, -1)
                pad = nn.ReflectionPad2d(win.shape[3]//2)
                f = F.conv2d(pad(f), win, stride=2, groups=win.shape[0])              
                if not f.shape[2] == feats[j+1].shape[2] or not f.shape[3] == feats[j+1].shape[3]:
                    f = F.interpolate(f, [feats[j+1].shape[2], feats[j+1].shape[3]], mode='bilinear', align_corners=True)
                f_pyr[j+1].append(f)

        for i in range(len(feats)):
            f_pyr[i] = torch.cat(f_pyr[i],dim=1)
        
        return f_pyr
       
    def forward(self, x,scale=5):
        # with torch.no_grad():
        feats_x = self.get_features(x,scale=scale)
        # feats_x = self.construct_gau_pyramid(feats_x)
        
        # feats_gau_x = gaussian_pyramid(x, max_levels=len(feats_x))
        
        return feats_x
    
    
def get_feature(path,vgg,scale=5):
    '''
    @description: get the feature of the image in the last layer of VGG
    @param {path} image path
    @return: the feature of the image as a Torch tensor
    '''
    if scale == 0:
        return get_img_pixel(path)
    # read the image using cv2 and convert it to a Torch tensor
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert the image to a Torch tensor
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    # get the feature of the image
    features = vgg(img,scale=scale)
    # concatenate the features of the image all together
    features = torch.cat(features,dim=1)
    return features

def get_img_pixel(path):
    '''
    @description: get the pixel value of the image
    @param {path} image path
    @return: the pixel value of the image as a Torch tensor
    '''
    # read the image using cv2 and convert it to a Torch tensor
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # convert the image to a Torch tensor
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)
    return img
