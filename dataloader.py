import  numpy as np
import  torch
import torch.nn as nn
import torchvision
from dataset.tid2013 import Tid2013Dataset
from dataset.tid2013 import BASE_PATH





class Preprocess(object):

    def __init__(self):
        pass


    def normalize_img(self,img):
        # to grayscale
        img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

        # subtract the low-pass img
        # @todo implemention

        return img

    def get_distored_img(self):
        pass

    def get_error_map(self):
        pass


    def __call__(self, dataItem):#@todo key of dataItem
        distored_img = dataItem['']
        reference_img = dataItem['']















