import  torch
import torch.nn as nn
import numpy as np




class DeepQANet(nn.Module):

    def __init__(self):
        super(DeepQANet, self).__init__()

        self.input_channel=1  #@todo
        self.num_ch=1

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


    def forward(self,distored_img,error_map):
        output_distored_img=self.distored_img_net(distored_img)
        output_error_map=self.error_map_net(error_map)

        output_total=torch.cat(output_distored_img,output_error_map,dim=0)
        output_total=self.sense_map_net(output_total)


    def forward2(self,r_patch_set,d_patch_set,me_set):
        pass
