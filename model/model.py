import  torch
import torch.nn as nn
import numpy as np





class DeepQANet(nn.Module):

    def __init__(self,device):
        super(DeepQANet, self).__init__()

        self.input_channel=1
        self.num_ch=1
        self.ign=4
        self.wl_subj = float(1e3)
        self.wr_l2 = float( 5e-3)
        self.wr_tv = float( 1e-2)
        self.device=device



        self.distored_img_net=nn.Sequential(
            nn.Conv2d(self.input_channel,32,kernel_size=3,stride=1,padding=(1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=(1,1)),
            nn.LeakyReLU(inplace=True),
        )

        self.error_map_net=nn.Sequential(
            nn.Conv2d(self.input_channel,32,kernel_size=3,stride=1,padding=(1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=(1,1)),
            nn.LeakyReLU(inplace=True),

        )

        last_conv= nn.Conv2d(64, self.num_ch, kernel_size=3, stride=1,padding=(1,1))
        last_conv.bias.data.fill_(1.0)

        self.sense_map_net=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=(1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2,padding=(1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=(1,1)),
            nn.LeakyReLU(inplace=True),
            last_conv,
            nn.ReLU(inplace=True),
        )


        self.regression_net=nn.Sequential(
            nn.Linear(self.num_ch,4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4 ,1),
            nn.ReLU(inplace=True)
            )



        def get_downsample_filter():

            downsample_filter = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(2, 2),padding=(2,2))

            k = np.float32([1, 4, 6, 4, 1])
            k = np.outer(k, k)
            k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
            downsample_weight = torch.from_numpy(k5x5)
            downsample_filter.weight = torch.nn.Parameter(downsample_weight)
            downsample_filter.bias.data.fill_(0)
            downsample_filter.weight.requires_grad=False
            downsample_filter.bias.requires_grad = False
            return downsample_filter
        self.downsample_filter=get_downsample_filter()



        def get_sobel_y_filter():
            sobel_y_filter = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1,padding=(1,1))

            sobel_y_val = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   dtype='float32').reshape((1, 1, 3, 3))
            sobel_y_filter_weight = torch.from_numpy(sobel_y_val)
            sobel_y_filter.weight = torch.nn.Parameter(sobel_y_filter_weight)
            sobel_y_filter.bias.data.fill_(0)
            sobel_y_filter.weight.requires_grad=False
            sobel_y_filter.bias.requires_grad = False

            return sobel_y_filter

        self.sobel_y_filter=get_sobel_y_filter()


        def get_sobel_x_filter():
            sobel_x_filter = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1,padding=(1,1))

            sobel_x_val = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                                   dtype='float32').reshape((1, 1, 3, 3))
            sobel_x_filter_weight = torch.from_numpy(sobel_x_val)
            sobel_x_filter.weight = torch.nn.Parameter(sobel_x_filter_weight)
            sobel_x_filter.bias.data.fill_(0)
            sobel_x_filter.weight.requires_grad=False
            sobel_x_filter.bias.requires_grad = False

            return sobel_x_filter

        self.sobel_x_filter=get_sobel_x_filter()

        def get_upsample_filter():
            upsample_filter = nn.ConvTranspose2d(1, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                                                 output_padding=(1, 1), bias=False)

            k = np.float32([1, 4, 6, 4, 1])
            k = np.outer(k, k)
            k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
            k5x5 *= 4
            upsample_weight = torch.from_numpy(k5x5)
            upsample_filter.weight = torch.nn.Parameter(upsample_weight)
            upsample_filter.weight.requires_grad = False

            return upsample_filter

        self.upsample_filter = get_upsample_filter()



        def get_log_diff_fn(eps=1.0):
            log_255_sq = np.float32(2 * np.log(255.0))
            log_255_sq = log_255_sq.item()#int
            max_val = np.float32(log_255_sq - np.log(eps))
            max_val = max_val.item()#int

            log_255_sq = torch.from_numpy(np.array(log_255_sq)).float().to(self.device)
            max_val=torch.from_numpy(np.array(max_val)).float().to(self.device)
            def log_diff_fn(in_a, in_b):
                diff = 255.0 * (in_a - in_b)
                val = log_255_sq - torch.log(diff ** 2 + eps)
                return val / max_val
            return log_diff_fn


        self.log_diff_fn=get_log_diff_fn(1.0)





    def forward_sens_map(self,distored_img,error_map):
        output_distored_img=self.distored_img_net(distored_img)
        output_error_map=self.error_map_net(error_map)

        output_total=torch.cat([output_distored_img,output_error_map],dim=1)
        output_total=self.sense_map_net(output_total)

        return output_total





    # this is the code for single patch (patch,channel,width,height)
    def forward(self,r_patch_set,d_patch_set,mos_set):



        d_patch_set_norm=self.normalize_lowpass_subt(d_patch_set,3,self.num_ch)
        error_map = self.log_diff_fn(r_patch_set, d_patch_set)
        e_ds4 = self.downsample_filter(self.downsample_filter(error_map))
        sense_map = self.forward_sens_map(d_patch_set_norm ,error_map)


        percep_map=sense_map * e_ds4
        percep_map_crop= self.shave_border(percep_map)



        mos_p=torch.mean(percep_map_crop,dim=(0,2,3),keepdim=True)
        mos_p=mos_p.reshape(mos_set.shape)



        ################################################
        #loss


        #mse loss
        subj_loss = self.get_mse(mos_p, mos_set)

        # TV norm regularization
        tv = self.get_total_variation(percep_map, 3.0)

        # l2 loss
        l2_reg = self.get_l2_regularization(
            [self.distored_img_net,self.error_map_net,self.sense_map_net,self.regression_net], mode='sum')


        total_loss=subj_loss*self.wl_subj+tv*self.wr_tv+l2_reg*self.wr_l2





        return total_loss,mos_p






    def get_l2_regularization(self, nets, mode='sum'):

        l2 = []
        if mode == 'sum':

            for net in nets:
                for key,layer in net._modules.items():
                    if hasattr(layer,'weight'):
                        l2.append(torch.sum(layer.weight**2).reshape(1))
                    if hasattr(layer,'bias'):
                        l2.append(torch.sum(layer.bias ** 2).reshape(1))

            l2 = torch.cat(l2)
            l2 = torch.sum(l2)

            return l2



    def get_mse(self, x, y, return_map=False):
        if return_map:
            return (x - y) ** 2
        else:
            return torch.mean((x - y) ** 2)


    def shave_border(self, feat_map):
        if self.ign > 0:
            return feat_map[:, :, self.ign:-self.ign, self.ign:-self.ign]
        else:
            return feat_map



    def get_total_variation(self, input, beta=3.0):
        """
        Calculate total variation of the input.
        Arguments
            x: 4D tensor image. It must have 1 channel feauture
        """
        x_grad = self.sobel_x_filter(input)
        y_grad =self.sobel_y_filter(input)
        tv = torch.mean((y_grad ** 2 + x_grad ** 2) ** (beta / 2))
        return tv

    def normalize_lowpass_subt(self,img, n_level, n_ch=1):
        '''Normalize image by subtracting the low-pass-filtered image'''
        # Downsample
        img_ = img
        pyr_sh = []
        for i in range(n_level - 1):
            pyr_sh.append(img_.shape)
            img_ = self.downsample_filter(img_)

        # Upsample
        for i in range(n_level - 1):
            img_ = self.upsample_filter(img_)
        return img - img_




