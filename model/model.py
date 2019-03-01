import  torch
import torch.nn as nn
import numpy as np
from  utils import downsample_img




class DeepQANet(nn.Module):

    def __init__(self):
        super(DeepQANet, self).__init__()

        self.input_channel=1 #@todo
        self.num_ch=1
        self.ign=4

        self.distored_img_net=nn.Sequential(
            nn.Conv2d(self.input_channel,32,kernel_size=3,stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
        )

        self.error_map_net=nn.Sequential(
            nn.Conv2d(self.input_channel,32,kernel_size=3,stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),

        )

        self.sense_map_net=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, self.num_ch, kernel_size=3, stride=1,bias=np.ones((self.num_ch,),dtype='float32') ),
            nn.ReLU(inplace=True),
        )


        self.regression_net=nn.Sequential(
            nn.Linear(self.num_ch,4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4 ,1),
            nn.ReLU(inplace=True)
            )


    def forward_sens_map(self,distored_img,error_map):
        output_distored_img=self.distored_img_net(distored_img)
        output_error_map=self.error_map_net(error_map)

        output_total=torch.cat(output_distored_img,output_error_map,dim=0)
        output_total=self.sense_map_net(output_total)

        return output_total


    def forward(self,r_patch_set,d_patch_set):
        #@todo to normalize_lowpass_subtract

        error_map=self.log_diff_fn(r_patch_set,d_patch_set,1.0)
        sense_map = self.forward_sens_map(d_patch_set, error_map)
        e_ds4=downsample_img(downsample_img(error_map, self.num_ch), self.num_ch)



        predict_map= sense_map * e_ds4
        predict_crop=self.shave_border(predict_map)

        #@todo to generate feature vector

        predict_mos=self.regression_net(predict_crop)

        return predict_mos



    def shave_border(self, feat_map):
        if self.ign > 0:
            return feat_map[:, :, self.ign:-self.ign, self.ign:-self.ign]
        else:
            return feat_map


    def log_diff_fn(self, in_a, in_b, eps=0.1):
        diff = 255.0 * (in_a - in_b)
        log_255_sq = np.float32(2 * np.log(255.0))

        val = log_255_sq - torch.log(diff ** 2 + eps)
        max_val = np.float32(log_255_sq - np.log(eps))
        return val / max_val