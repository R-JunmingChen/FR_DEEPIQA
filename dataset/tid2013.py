import numpy as np
from torch.utils.data.dataset import Dataset
import PIL.Image as Image
import torch
import os
import random
import logging
from scipy import misc
from scipy.ndimage.filters import convolve


# Define DB information
BASE_PATH = '/Users/mayzha//datasets/IQA/tid2013'
LIST_FILE_NAME = '../TID2013.txt'
TRAIN_RATIO=0.8
ALL_SCENES = list(range(24))
# ALL_SCENES = list(range(25))
ALL_DIST_TYPES = list(range(24))


logger=logging.getLogger('IQA')



def get_dataset():
    """
           Make train and test image list from TID2013 database
           TID2013: 25 reference images x 24 distortions x 5 levels
           """
    logger.info("start to load dataset info")

    scenes, dist_types, d_img_list, r_img_list, score_list = [], [], [], [], []
    list_file_name = LIST_FILE_NAME
    with open(list_file_name, 'r') as listFile:
        for line in listFile:
            (scn_idx, dis_idx, ref, dis, score) = line.split()
            scn_idx = int(scn_idx)
            dis_idx = int(dis_idx)

            scenes.append(scn_idx)
            dist_types.append(dis_idx)
            r_img_list.append(ref)
            d_img_list.append(dis)
            score_list.append(float(score))

    n_images = len(d_img_list)

    # divide scene randomly get train and test dataset
    random.shuffle(ALL_SCENES)
    train_scenes_indexs = ALL_SCENES[:int(TRAIN_RATIO * len(ALL_SCENES))]
    test_scenes_indexs = ALL_SCENES[int(TRAIN_RATIO * len(ALL_SCENES)):len(ALL_SCENES)]

    train_scenes, train_dist_types, train_d_img_list, train_r_img_list, train_score_list = [], [], [], [], []
    test_scenes, test_dist_types, test_d_img_list, test_r_img_list, test_score_list = [], [], [], [], []
    for index in range(n_images):
        if scenes[index] in train_scenes_indexs:  # train
            train_scenes.append(scenes[index])
            train_dist_types.append(dist_types[index])
            train_d_img_list.append(d_img_list[index])
            train_r_img_list.append(r_img_list[index])
            train_score_list.append(score_list[index])
        else:
            test_scenes.append(scenes[index])  # test
            test_dist_types.append(dist_types[index])
            test_d_img_list.append(d_img_list[index])
            test_r_img_list.append(r_img_list[index])
            test_score_list.append(score_list[index])

    train_score_list = np.array(train_score_list, dtype='float32')
    test_score_list = np.array(test_score_list, dtype='float32')

    dataSetInfo = {
        'train': {
            'scenes': train_scenes,  # int list
            'dist_types': train_dist_types,  # int list
            'r_img_path_list': train_r_img_list,  # string list
            'd_img_path_list': train_d_img_list,  # string list
            'score_list': train_score_list,  # numpy array
            'n_images': len(train_r_img_list),  # int
            'dataset_dir': BASE_PATH,  # string
        },
        'train': {
            'scenes': test_scenes,  # int list
            'dist_types': test_dist_types,  # int list
            'r_img_path_list': test_r_img_list,  # string list
            'd_img_path_list': test_d_img_list,  # string list
            'score_list': test_score_list,  # numpy array
            'n_images': len(test_r_img_list),  # int
            'dataset_dir': BASE_PATH,  # string
        },

    }


    return  dataSetInfo

dataSetInfo=get_dataset()


class Tid2013Dataset(Dataset):

    def __init__(self,dataset_dir,transform,type='train'):

        self.dataSetInfo=dataSetInfo[type]

        self.dataset_dir=dataset_dir
        self.transform=transform #@todo
        self.color='gray'

        self.patch_step=[80,80]
        self.patch_size=[112,112]

        self.patch_mode = 'shift_center'
        self.local_norm= False


        self.std_filt_r=1.0
        self.fr_met=None
        self.fr_met_avg=False
        self.num_ch=1
        self.fr_met_scale=1.0
        self.random_crops=0

        


    def __len__(self):
        return self.dataSetInfo['train']['n_images']+self.dataSetInfo['test']['n_images']




    def load_reference_img(self,img_relative_path):
        """
        load single reference_img to patches
        """

        patch_size=self.patch_size
        patch_step=self.patch_step


        ref_top_left_set = []
        r_pat_set = []
        pass_list = []

        # img path
        img_path=os.path.join(self.dataset_dir,img_relative_path)
        # open with scipy misc
        img = misc.imread(img_path)
        current_h=img.shape[0]
        current_w=img.shape[1]

        # Gray
        img = convert_color2(img, self.color)

        # Local normalization
        if self.local_norm:
            if self.color == 'gray':
                # faster
                r_img_norm = local_normalize_1ch(img)
            else:
                r_img_norm = local_normalize(img, self.num_ch)
        else:
            r_img_norm = img.astype('float32') / 255.

        if self.color == 'gray':
            r_img_norm = r_img_norm[:, :, None]


        # numbers of patches along y and x axes
        ny = (current_h - patch_size[0]) // patch_step[0] + 1
        nx = (current_w - patch_size[1]) // patch_step[1] + 1
        patch_info=(int(ny*nx),ny,nx)



        # get non-covered length along y and x axes
        cov_height = patch_step[0] * (ny - 1) + patch_size[0]
        cov_width = patch_step[1] * (nx - 1) + patch_size[1]
        nc_height = current_h - cov_height
        nc_width = current_w - cov_width

        # Shift center
        if self.patch_mode == 'shift_center':
            shift = [(nc_height + 1) // 2, (nc_width + 1) // 2]
            if shift[0] % 2 == 1:
                shift[0] -= 1
            if shift[1] % 2 == 1:
                shift[1] -= 1
            shift = tuple(shift)
        else:
            shift = (0, 0)

        # generate top_left_set of patches
        top_left_set = np.zeros((nx * ny, 2), dtype=np.int)
        for yidx in range(ny):
            for xidx in range(nx):
                top = (yidx * patch_step[0] + shift[0])
                left = (xidx * patch_step[1] + shift[1])
                top_left_set[yidx * nx + xidx] = [top, left]
        ref_top_left_set.append(top_left_set)

        # Crop the images to patches
        for idx in range(ny * nx):
            [top, left] = top_left_set[idx]

            # if top + patch_size[0] > current_h:
            #
            #     print(' (%d > %d)' % (top + patch_size[0], current_h))
            #
            # if left + patch_size[1] > current_w:
            #
            #     print(' (%d > %d)' % (left + patch_size[1], current_w))

            r_crop_norm = r_img_norm[top:top + patch_size[0],
                                     left:left + patch_size[1]]

            r_pat_set.append(r_crop_norm)

        # if len(pass_list) > 0:
        #     self.n_images -= len(pass_list)
        #     print(' - Ignored ref. images due to small size: %s' %
        #           ', '.join(str(i) for i in pass_list))


        return patch_info,ref_top_left_set, r_pat_set











    def load_distored_img(self,patch_info,ref_top_left_set,img_relative_path):
        img_path=os.path.join(self.dataset_dir,img_relative_path)




        patch_size = self.patch_size

        n_patches = 0
        npat_img_list = []
        d_pat_set = []
        loc_met_set = []
        filt_idx_list = []
        dis2ref_idx = []


        pat_idx = 0
        pass_list = []

        # Read ref. and dist. images
        d_img_raw = misc.imread(img_path)

        cur_h = d_img_raw.shape[0]
        cur_w = d_img_raw.shape[1]

        # Gray or RGB
        d_img = convert_color2(d_img_raw, self.color)

        # Read local metric scores
        if self.fr_met:
            ext = int(1. / self.fr_met_scale) - 1
            met_size = (int((cur_h + ext) * self.fr_met_scale),
                        int((cur_w + ext) * self.fr_met_scale))
            met_pat_size = (int((patch_size[0] + ext) * self.fr_met_scale),
                            int((patch_size[1] + ext) * self.fr_met_scale))
            if self.fr_met == 'SSIM_now':
                # d_img_ds = misc.imresize(d_img, met_size, interp='bicubic')
                # r_img_ds = misc.imresize(r_img, met_size, interp='bicubic')
                # loc_q_map = ssim(d_img_ds, r_img_ds)
                raise NotImplementedError()
            else:
                met_s_fname = (img_relative_path +
                               self.fr_met_suffix + self.fr_met_ext)
                loc_q_map = np.fromfile(
                    os.path.join(self.fr_met_path, self.fr_met_subpath,
                                 met_s_fname),
                    dtype='float32')
                loc_q_map = loc_q_map.reshape(
                    (met_size[1], met_size[0])).transpose()

        # Local normalization
        if self.local_norm:
            if self.color == 'gray':
                # faster
                d_img_norm = local_normalize_1ch(d_img)
            else:
                d_img_norm = local_normalize(d_img, self.num_ch)
        else:
            d_img_norm = d_img.astype('float32') / 255.

        if self.color == 'gray':
            d_img_norm = d_img_norm[:, :, None]

        top_left_set = ref_top_left_set

        if np.array(top_left_set).shape[0]==1:
           top_left_set=ref_top_left_set[0]

        cur_n_patches=np.array(top_left_set).shape[0]





        if self.random_crops > 0:
            if self.random_crops < cur_n_patches:
                n_crops = self.random_crops
                rand_perm = np.random.permutation(cur_n_patches)
                sel_patch_idx = sorted(rand_perm[:n_crops])
                top_left_set = top_left_set[sel_patch_idx].copy()
            else:
                n_crops = cur_n_patches
                sel_patch_idx = np.arange(cur_n_patches)

            npat_filt = n_crops
            npat_img_list.append((npat_filt, 1, npat_filt))
            n_patches += npat_filt

            idx_set = list(range(npat_filt))
            filt_idx_list.append(idx_set)

        else:
            # numbers of patches along y and x axes
            npat, ny, nx = patch_info
            npat_filt = int(npat * self.std_filt_r)

            npat_img_list.append((npat_filt, ny, nx))
            n_patches += npat_filt

            if self.std_filt_r < 1.0:
                std_set = np.zeros((nx * ny))
                for idx, top_left in enumerate(top_left_set):
                    top, left = top_left
                    std_set[idx] = np.std(
                        d_img[top:top + patch_size[0],
                              left:left + patch_size[1]])

            # Filter the patches with low std
            if self.std_filt_r < 1.0:
                idx_set = sorted(list(range(len(std_set))),
                                 key=lambda x: std_set[x], reverse=True)
                idx_set = sorted(idx_set[:npat_filt])
            else:
                idx_set = list(range(npat_filt))
            filt_idx_list.append(idx_set)

        # Crop the images to patches
        for idx in idx_set:
            [top, left] = top_left_set[idx]


            # if top + patch_size[0] > cur_h:
            #     print('\n@Error: imidx=%d, pat=%d' % (im_idx, idx), end='')
            #     print(' (%d > %d)' % (top + patch_size[0], cur_h))
            #
            # if left + patch_size[1] > cur_w:
            #     print('\n@Error: imidx=%d, pat=%d' % (im_idx, idx), end='')
            #     print(' (%d > %d)' % (left + patch_size[1], cur_w))

            d_crop_norm = d_img_norm[top:top + patch_size[0],
                                     left:left + patch_size[1]]

            d_pat_set.append(d_crop_norm)

            # if self.random_crops > 0:
            #     dis2ref_idx.append(
            #         ref_img2pat_idx[ref_idx][sel_patch_idx[idx]])
            # else:
            #     dis2ref_idx.append(ref_img2pat_idx[ref_idx][idx])

            # Crop the local metric scores
            if self.fr_met:
                ext = int(1. / self.fr_met_scale) - 1
                top_r = int((top + ext) * self.fr_met_scale)
                left_r = int((left + ext) * self.fr_met_scale)

                # if top_r + met_pat_size[0] > met_size[0]:
                #     print('\n@Error (FR metric size):', end='')
                #     print(' imidx=%d, pat=%d' % (im_idx, idx), end='')
                #     print(' (%d > %d)' % (
                #         top_r + met_pat_size[0], met_size[0]))
                #
                # if left_r + met_pat_size[1] > met_size[1]:
                #     print('\n@Error (FR metric size):', end='')
                #     print(' imidx=%d, pat=%d' % (im_idx, idx), end='')
                #     print(' (%d > %d)' % (
                #         left_r + met_pat_size[1], met_size[1]))

                loc_met_crop = loc_q_map[top_r:top_r + met_pat_size[0],
                                         left_r:left_r + met_pat_size[1]]
                # if loc_met_crop.shape != met_pat_size:
                    # print('\n@Error (oc_met_crop.shape != met_pat_size)')
                    # print("@ image (%d-%d):" % (im_idx, idx),
                    #       d_img_list[im_idx])
                    # print("@ loc_met_crop.shape:", loc_met_crop.shape)
                    # print("@ met_size:", met_size)
                    # print("@ top_r:", top_r)
                    # print("@ left_r:", left_r)
                    # os.system("pause")

                if self.fr_met_avg:
                    loc_met_set.append(
                        np.mean(loc_met_crop, keepdims=True))
                else:
                    loc_met_set.append(loc_met_crop)

            pat_idx += 1



        # if len(pass_list) > 0:
        #     self.n_images -= len(pass_list)
        #     print(' - Ignored image list due to small size: %s' %
        #           ', '.join(str(i) for i in pass_list))

        self.n_patches = n_patches
        self.npat_img_list = npat_img_list
        self.d_pat_set = d_pat_set
        if self.fr_met:
            self.loc_met_set = loc_met_set
        self.filt_idx_list = filt_idx_list
        self.dis2ref_idx = dis2ref_idx


        return d_pat_set





    def __getitem__(self, index):
        patch_info,ref_top_left_set, r_pat_set=self.load_reference_img(self.dataSetInfo['r_img_path_list'][index])
        d_pat_set=self.load_distored_img(patch_info,ref_top_left_set,self.dataSetInfo['d_img_path_list'][index])
        mos= self.dataSetInfo['score_list'][index]

        return r_pat_set,d_pat_set,mos




        r_img_path=os.path.join(self.dataset_dir,self.dataSetInfo['r_img_path_list'][index])
        d_img_path=os.path.join(self.dataset_dir,self.dataSetInfo['d_img_path_list'][index])

        r_img=Image.open(r_img_path)
        d_img=Image.open(d_img_path)
        
        
        #trans to np array and transpose to CWH
        r_img=np.asanyarray(r_img).transpose(-1, 0,
                                        1)  # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)

        d_img = np.asanyarray(d_img).transpose(-1, 0,
                                               1)
        
        
        #preprocess
        
        
        
        
        
        #trans to torch tensor








        # img = Image.open(self.x[index])  # use pillow to open a file
        # img = img.resize((self.width, self.height))  # resize the file to 256x256
        # img = img.convert('RGB')  # convert image to RGB channel
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # img = np.asarray(img).transpose(-1, 0,
        #                                 1)  # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
        # img = torch.from_numpy(np.asarray(img))  # create the image tensor
        # label = torch.from_numpy(np.asarray(self.y[index]).reshape([1, 1]))  # create the label tensor
        # return img, label




def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, :] = im[:, :, np.newaxis]
    return ret


def rgb2gray(rgb):
    assert rgb.shape[2] == 3
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def rgb2ycbcr(rgb):
    xform = np.array([[.299, .587, .114],
                      [-.1687, -.3313, .5],
                      [.5, -.4187, -.0813]])
    ycbcr = np.dot(rgb[..., :3], xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def ycbcr2rgb(ycbcr):
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -.71414],
                      [1, 1.772, 0]])
    rgb = ycbcr.astype('float32')
    rgb[:, :, [1, 2]] -= 128
    return rgb.dot(xform.T)


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
kern = k / k.sum()


def local_normalize_1ch(img, const=127.0):
    mu = convolve(img, kern, mode='nearest')
    mu_sq = mu * mu
    im_sq = img * img
    tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
    sigma = np.sqrt(np.abs(tmp))
    structdis = (img - mu) / (sigma + const)

    # Rescale within 0 and 1
    # structdis = (structdis + 3) / 6
    structdis = 2. * structdis / 3.
    return structdis


def local_normalize(img, num_ch=1, const=127.0):
    if num_ch == 1:
        mu = convolve(img[:, :, 0], kern, mode='nearest')
        mu_sq = mu * mu
        im_sq = img[:, :, 0] * img[:, :, 0]
        tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
        sigma = np.sqrt(np.abs(tmp))
        structdis = (img[:, :, 0] - mu) / (sigma + const)

        # Rescale within 0 and 1
        # structdis = (structdis + 3) / 6
        structdis = 2. * structdis / 3.
        norm = structdis[:, :, None]
    elif num_ch > 1:
        norm = np.zeros(img.shape, dtype='float32')
        for ch in range(num_ch):
            mu = convolve(img[:, :, ch], kern, mode='nearest')
            mu_sq = mu * mu
            im_sq = img[:, :, ch] * img[:, :, ch]
            tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
            sigma = np.sqrt(np.abs(tmp))
            structdis = (img[:, :, ch] - mu) / (sigma + const)

            # Rescale within 0 and 1
            # structdis = (structdis + 3) / 6
            structdis = 2. * structdis / 3.
            norm[:, :, ch] = structdis

    return norm



def convert_color2(img, color):
    """ Convert image into gray or RGB or YCbCr.
    (In case of gray, dimension is not increased for
    the faster local normalization.)
    """
    assert len(img.shape) in [2, 3]
    if color == 'gray':
        # if d_img_raw.shape[2] == 1:
        if len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2gray(img)
    elif color == 'rgb':
        if len(img.shape) == 2:  # if gray
            img_ = gray2rgb(img)
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = img
    elif color == 'ycbcr':
        if len(img.shape) == 2:  # if gray
            img_ = rgb2ycbcr(gray2rgb(img))
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2ycbcr(img)
    else:
        raise ValueError("Improper color selection: %s" % color)

    return img_


if __name__ == '__main__':
    dataset=Tid2013Dataset("/Users/mayzha//datasets/IQA/tid2013",None)

    r_pat_set,d_pat_set,loc_met_crop=dataset[1]