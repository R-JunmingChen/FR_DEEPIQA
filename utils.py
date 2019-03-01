import  numpy as np
import  torch
from   scipy import convolve
import torch.nn as nn

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = (k / k.sum()).reshape((1, 1, 5, 5))
kern = k5x5 #theano.shared(k5x5, borrow=True)  #@todo theano shared

k5x5_3ch = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
k5x5_3ch = k5x5_3ch.transpose([2, 3, 0, 1])
kern_3ch = k5x5_3ch #theano.shared(k5x5_3ch, borrow=True) #@todo theano shared


def log_diff_fn(self, in_a, in_b, eps=1.0):
    """
    get the error map by log difference function
    """
    diff = 255.0 * (in_a - in_b)
    log_255_sq = np.float32(2 * np.log(255.0))

    val = log_255_sq - T.log(diff ** 2 + eps)
    max_val = np.float32(log_255_sq - np.log(eps))
    return val / max_val

def downsample_img(img, n_ch=1):
    if n_ch == 1:
        kernel = kern
        filter_shape = [1, 1, 5, 5]
    elif n_ch == 3:
        kernel = kern_3ch
        filter_shape = [3, 3, 5, 5]
    else:
        raise NotImplementedError
    return nn.Conv2d(img, kernel, kernel_size=filter_shape,
                   stride=(2, 2))  #@todo unkown the meaning of border_mode='half' in theano
#
# def conv2d_tr_half(output, filters, filter_shape, input_shape,
#                    subsample=(1, 1)):
#     input = conv2d_grad_wrt_inputs(              #@todo this func is no implmention in pytorch
#         output, filters,
#         input_shape=(None, filter_shape[0], input_shape[2], input_shape[3]),
#         filter_shape=filter_shape, border_mode='half', subsample=subsample)  #@todo unkown the meaning of border_mode='half' in theano
#     return input
#
# def upsample_img(img, out_shape, n_ch=1):
#     if n_ch == 1:
#         kernel = kern * 4
#         filter_shape = [1, 1, 5, 5]
#     elif n_ch == 3:
#         kernel = kern_3ch * 4
#         filter_shape = [3, 3, 5, 5]
#     else:
#         raise NotImplementedError
#     return conv2d_tr_half(img, kernel, filter_shape=filter_shape,
#                           input_shape=out_shape, subsample=(2, 2))


def normalize_lowpass_subt(img, n_level, n_ch=1):
    '''Normalize image by subtracting the low-pass-filtered image'''
    # Downsample
    img_ = img
    pyr_sh = []
    for i in range(n_level - 1):
        pyr_sh.append(img_.shape)
        img_ = downsample_img(img_, n_ch)

    # Upsample
    for i in range(n_level - 1):
        img_ = upsample_img(img_, pyr_sh[n_level - 2 - i], n_ch)
    return img - img_



def show_progress(percent):
    hashes = '#' * int(round(percent * 20))
    spaces = ' ' * (20 - len(hashes))
    sys.stdout.write("\r - Load images: [{0}] {1}%".format(
        hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def convert_color(img, color):
    """ Convert image into gray or RGB or YCbCr.
    """
    assert len(img.shape) in [2, 3]
    if color == 'gray':
        # if d_img_raw.shape[2] == 1:
        if len(img.shape) == 2:  # if gray
            img_ = img[:, :, None]
        elif len(img.shape) == 3:  # if RGB
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img_ = rgb2gray(img)[:, :, None]
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
