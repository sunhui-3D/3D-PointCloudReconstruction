# -*- coding: utf-8 -*-
import cv2

class Compose(object):
    """ Composes several transforms together.
        For example:
      ##  >>> transforms.Compose([
      #  >>>     transforms.RandomBackground(),
      #  >>>     transforms.CenterCrop(127, 127, 3),
#        >>>  ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rendering_images, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                rendering_images = t(rendering_images, bounding_box)
            else:
                rendering_images = t(rendering_images)

        return rendering_images


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        # array = np.transpose(rendering_images, (0, 3, 1, 2))
        # handle numpy array
        tensor = torch.from_numpy(rendering_images)

        # put it from HWC to CHW format
        print('ToTensor ', tensor.shape)
        return tensor.float()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, rendering_images):
        # print('noramlizerendering_images', rendering_images.shape)
        assert (isinstance(rendering_images, np.ndarray))
        rendering_images -= self.mean
        rendering_images /= self.std
        print('normalize', rendering_images.shape)

        return rendering_images


class RandomPermuteRGB(object):
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))

        random_permutation = np.random.permutation(3)
        for img_idx, img in enumerate(rendering_images):
            rendering_images[img_idx] = img[..., random_permutation]

        return rendering_images


class CenterCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, rendering_images, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images

        crop_size_c = rendering_images[0].shape[2]
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if bounding_box is not None:
                bounding_box = [
                    bounding_box[0] * img_width,
                    bounding_box[1] * img_height,
                    bounding_box[2] * img_width,
                    bounding_box[3] * img_height
                ]  # yapf: disable

                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

                # Make the crop area as a square
                square_object_size = max(bbox_width, bbox_height)
                x_left = int(bbox_x_mid - square_object_size * .5)
                x_right = int(bbox_x_mid + square_object_size * .5)
                y_top = int(bbox_y_mid - square_object_size * .5)
                y_bottom = int(bbox_y_mid + square_object_size * .5)

                # If the crop position is out of the image, fix it with padding
                pad_x_left = 0
                if x_left < 0:
                    pad_x_left = -x_left
                    x_left = 0
                pad_x_right = 0
                if x_right >= img_width:
                    pad_x_right = x_right - img_width + 1
                    x_right = img_width - 1
                pad_y_top = 0
                if y_top < 0:
                    pad_y_top = -y_top
                    y_top = 0
                pad_y_bottom = 0
                if y_bottom >= img_height:
                    pad_y_bottom = y_bottom - img_height + 1
                    y_bottom = img_height - 1

                # Padding the image and resize the image
                processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                         ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                         mode='edge')
                processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = int(img_width - self.crop_size_w) // 2
                    x_right = int(x_left + self.crop_size_w)
                    y_top = int(img_height - self.crop_size_h) // 2
                    y_bottom = int(y_top + self.crop_size_h)
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

                processed_image = cv2.resize(img[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))

            processed_images = np.append(processed_images, [processed_image], axis=0)
            # Debug
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(img)
            # if not bounding_box is None:
            #     rect = patches.Rectangle((bounding_box[0], bounding_box[1]),
            #                              bbox_width,
            #                              bbox_height,
            #                              linewidth=1,
            #                              edgecolor='r',
            #                              facecolor='none')
            #     ax1.add_patch(rect)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image)
            # plt.show()
        return processed_images


class RandomCrop(object):
    def __init__(self, img_size, crop_size):
        """Set the height and weight before and after cropping"""
        self.img_size_h = img_size[0]
        self.img_size_w = img_size[1]
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, rendering_images, bounding_box=None):
        if len(rendering_images) == 0:
            return rendering_images

        crop_size_c = rendering_images[0].shape[2]
        processed_images = np.empty(shape=(0, self.img_size_h, self.img_size_w, crop_size_c))
        for img_idx, img in enumerate(rendering_images):
            img_height, img_width, _ = img.shape

            if bounding_box is not None:
                bounding_box = [
                    bounding_box[0] * img_width,
                    bounding_box[1] * img_height,
                    bounding_box[2] * img_width,
                    bounding_box[3] * img_height
                ]  # yapf: disable

                # Calculate the size of bounding boxes
                bbox_width = bounding_box[2] - bounding_box[0]
                bbox_height = bounding_box[3] - bounding_box[1]
                bbox_x_mid = (bounding_box[2] + bounding_box[0]) * .5
                bbox_y_mid = (bounding_box[3] + bounding_box[1]) * .5

                # Make the crop area as a square
                square_object_size = max(bbox_width, bbox_height)
                square_object_size = square_object_size * random.uniform(0.8, 1.2)

                x_left = int(bbox_x_mid - square_object_size * random.uniform(.4, .6))
                x_right = int(bbox_x_mid + square_object_size * random.uniform(.4, .6))
                y_top = int(bbox_y_mid - square_object_size * random.uniform(.4, .6))
                y_bottom = int(bbox_y_mid + square_object_size * random.uniform(.4, .6))

                # If the crop position is out of the image, fix it with padding
                pad_x_left = 0
                if x_left < 0:
                    pad_x_left = -x_left
                    x_left = 0
                pad_x_right = 0
                if x_right >= img_width:
                    pad_x_right = x_right - img_width + 1
                    x_right = img_width - 1
                pad_y_top = 0
                if y_top < 0:
                    pad_y_top = -y_top
                    y_top = 0
                pad_y_bottom = 0
                if y_bottom >= img_height:
                    pad_y_bottom = y_bottom - img_height + 1
                    y_bottom = img_height - 1

                # Padding the image and resize the image
                processed_image = np.pad(img[y_top:y_bottom + 1, x_left:x_right + 1],
                                         ((pad_y_top, pad_y_bottom), (pad_x_left, pad_x_right), (0, 0)),
                                         mode='edge')
                processed_image = cv2.resize(processed_image, (self.img_size_w, self.img_size_h))
            else:
                if img_height > self.crop_size_h and img_width > self.crop_size_w:
                    x_left = int(img_width - self.crop_size_w) // 2
                    x_right = int(x_left + self.crop_size_w)
                    y_top = int(img_height - self.crop_size_h) // 2
                    y_bottom = int(y_top + self.crop_size_h)
                else:
                    x_left = 0
                    x_right = img_width
                    y_top = 0
                    y_bottom = img_height

                processed_image = cv2.resize(img[y_top:y_bottom, x_left:x_right], (self.img_size_w, self.img_size_h))

            processed_images = np.append(processed_images, [processed_image], axis=0)

        return processed_images


class RandomFlip(object):
    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))

        # for img_idx, img in enumerate(rendering_images):
        if random.randint(0, 1):
            rendering_images = np.fliplr(rendering_images)
        print('rendering_images', rendering_images.shape)

        return rendering_images


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, rendering_images):
        if len(rendering_images) == 0:
            return rendering_images

        # Allocate new space for storing processed images
        img_height, img_width, img_channels = rendering_images.shape
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels))
        # print('processed_images', processed_images.shape)

        # Randomize the value of changing brightness, contrast, and saturation
        brightness = 1 + np.random.uniform(low=-self.brightness, high=self.brightness)
        contrast = 1 + np.random.uniform(low=-self.contrast, high=self.contrast)
        saturation = 1 + np.random.uniform(low=-self.saturation, high=self.saturation)

        # Randomize the order of changing brightness, contrast, and saturation
        attr_names = ['brightness', 'contrast', 'saturation']
        attr_values = [brightness, contrast, saturation]    # The value of changing attrs
        attr_indexes = np.array(range(len(attr_names)))    # The order of changing attrs
        np.random.shuffle(attr_indexes)

        for img_idx, img in enumerate(rendering_images):
            # print('rendering_images', rendering_images.shape)
            # print('img_render', img.shape)
            processed_image = rendering_images
            for idx in attr_indexes:
                processed_image = self._adjust_image_attr(processed_image, attr_names[idx], attr_values[idx])

            processed_images = np.append(processed_images, [processed_image], axis=0)
            # print('ColorJitter', np.mean(ori_img), np.mean(processed_image))
            # fig = plt.figure(figsize=(8, 4))
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax1.imshow(ori_img)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax2.imshow(processed_image)
            # plt.show()
        return processed_images

    def _adjust_image_attr(self, img, attr_name, attr_value):
        """
        Adjust or randomize the specified attribute of the image

        Args:
            img: Image in BGR format
                Numpy array of shape (h, w, 3)
            attr_name: Image attribute to adjust or randomize
                       'brightness', 'saturation', or 'contrast'
            attr_value: the alpha for blending is randomly drawn from [1 - d, 1 + d]

        Returns:
            Output image in BGR format
            Numpy array of the same shape as input
        """
        # print('img', img.shape)
        gs = self._bgr_to_gray(img)

        if attr_name == 'contrast':
            img = self._alpha_blend(img, np.mean(gs[:, :, 0]), attr_value)
        elif attr_name == 'saturation':
            img = self._alpha_blend(img, gs, attr_value)
        elif attr_name == 'brightness':
            img = self._alpha_blend(img, 0, attr_value)
        else:
            raise NotImplementedError(attr_name)
        return img

    def _bgr_to_gray(self, bgr):
        """
        Convert a RGB image to a grayscale image
            Differences from cv2.cvtColor():
                1. Input image can be float
                2. Output image has three repeated channels, other than a single channel

        Args:
            bgr: Image in BGR format
                 Numpy array of shape (h, w, 3)

        Returns:
            gs: Grayscale image
                Numpy array of the same shape as input; the three channels are the same
        """
        ch = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
        gs = np.dstack((ch, ch, ch))
        return gs

    def _alpha_blend(self, im1, im2, alpha):
        """
        Alpha blending of two images or one image and a scalar

        Args:
            im1, im2: Image or scalar
                Numpy array and a scalar or two numpy arrays of the same shape
            alpha: Weight of im1
                Float ranging usually from 0 to 1

        Returns:
            im_blend: Blended image -- alpha * im1 + (1 - alpha) * im2
                Numpy array of the same shape as input image
        """
        im_blend = alpha * im1 + (1 - alpha) * im2
        return im_blend


class RandomNoise(object):
    def __init__(self,
                 noise_std,
                 eigvals=(0.2175, 0.0188, 0.0045),
                 eigvecs=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203))):
        self.noise_std = noise_std
        self.eigvals = np.array(eigvals)
        self.eigvecs = np.array(eigvecs)

    def __call__(self, rendering_images):
        alpha = np.random.normal(loc=0, scale=self.noise_std, size=3)
        noise_rgb = \
            np.sum(
                np.multiply(
                    np.multiply(
                        self.eigvecs,
                        np.tile(alpha, (3, 1))
                    ),
                    np.tile(self.eigvals, (3, 1))
                ),
                axis=1
            )
        print('noise_rgb', noise_rgb.shape)

        # Allocate new space for storing processed images
        img_height, img_width, img_channels = rendering_images.shape
        assert (img_channels == 3), "Please use RandomBackground to normalize image channels"
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels))

        # for img_idx, img in enumerate(rendering_images):
        processed_image = rendering_images[:, :, ::-1]    # BGR -> RGB
        # print('processed_image', processed_image.shape)
        for i in range(img_channels):
            processed_image[:, :, i] += noise_rgb[i]
        print(processed_image.shape)

        processed_image = processed_image[:, :, ::-1]    # RGB -> BGR
        processed_images = np.append(processed_images, [processed_image], axis=0)
        # from copy import deepcopy
        # ori_img = deepcopy(img)
        # print(noise_rgb, np.mean(processed_image), np.mean(ori_img))
        # print('RandomNoise', np.mean(ori_img), np.mean(processed_image))
        # fig = plt.figure(figsize=(8, 4))
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax1.imshow(ori_img)
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(processed_image)
        # plt.show()
        print('RandomNoise processed_image', processed_images.shape)
        return processed_images


class RandomBackground(object):
    def __init__(self, random_bg_color_range, random_bg_folder_path=None):
        self.random_bg_color_range = random_bg_color_range
        self.random_bg_files = []
        if random_bg_folder_path is not None:
            self.random_bg_files = os.listdir(random_bg_folder_path)
            self.random_bg_files = [os.path.join(random_bg_folder_path, rbf) for rbf in self.random_bg_files]

    def __call__(self, rendering_images):
        if len(rendering_images) == 0:
            return rendering_images
        # print('rendering_images', rendering_images.shape)

        img_height, img_width, img_channels  = rendering_images.shape
        # If the image has the alpha channel, add the background
        if not img_channels == 4:
            return rendering_images

        # Generate random background
        r, g, b = np.array([
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)
        ]) / 255.

        random_bg = None
        if len(self.random_bg_files) > 0:
            random_bg_file_path = random.choice(self.random_bg_files)
            random_bg = cv2.imread(random_bg_file_path).astype(np.float32) / 255.

        # Apply random background
        processed_images = np.empty(shape=(0, img_height, img_width, img_channels - 1))
        for img_idx, img in enumerate(rendering_images):
            alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
            img = img[:, :, :3]
            bg_color = random_bg if random.randint(0, 1) and random_bg is not None else np.array([[[r, g, b]]])
            img = alpha * bg_color + (1 - alpha) * img

            processed_images = np.append(processed_images, [img], axis=0)
        print('RandomBackground processed_images', processed_images.shape)

        return processed_images


import numpy as np
from PIL import Image


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img




    # my_transforms.AddSaltPepperNoise(0.2),
    # transforms.ToTensor(),
    #
    # my_transforms.AddGaussianNoise(mean=0, variance=1, amplitude=20),
    # transforms.ToTensor(),
    #
    #
    # transforms.RandomChoice(
    #     [transforms.RandomGrayscale(1), transforms.ColorJitter(hue=0.5), transforms.RandomRotation(60, resample=2)]),
    # transforms.ToTensor(),


import numpy as np
import random
from PIL import Image

# 自定义添加椒盐噪声的 transform
class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 如果随机概率小于 seld.p，则执行 transform
        if random.uniform(0, 1) < self.p:
            # 把 image 转为 array
            img_ = np.array(img).copy()
            # 获得 shape
            h, w, c = img_.shape
            # 信噪比
            signal_pct = self.snr
            # 椒盐噪声的比例 = 1 -信噪比
            noise_pct = (1 - self.snr)
            # 选择的值为 (0, 1, 2)，每个取值的概率分别为 [signal_pct, noise_pct/2., noise_pct/2.]
            # 椒噪声和盐噪声分别占 noise_pct 的一半
            # 1 为盐噪声，2 为 椒噪声
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            # 再转换为 image
            img_ = Image.fromarray(img_.astype(np.uint8))
            print(img_.mode)
            return img_.convert('RGB')#'uint8'
        # 如果随机概率大于 seld.p，则直接返回原图
        else:
            return img


import os
import numpy as np
import torch
import random
import math
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
# from enviroments import rmb_split_dir
# from lesson2.transforms.addPepperNoise import AddPepperNoise
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}

#对 tensor 进行反标准化操作，并且把 tensor 转换为 image，方便可视化。
def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """

    # 如果有标准化操作
    if 'Normalize' in str(transform_train):
        # 取出标准化的 transform
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        # 取出均值
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        # 取出标准差
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        # 乘以标准差，加上均值
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    # 把 C*H*W 变为 H*W*C
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    # 把 0~1 的值变为 0~255
    img_ = np.array(img_) * 255

    # 如果是 RGB 图
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
        # 如果是灰度图
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.RandomChoice([transforms.RandomVerticalFlip(p=0.8), transforms.RandomHorizontalFlip(p=0.8)]),

    # 3 RandomRotation
    transforms.RandomRotation(30),

    # 2 ColorJitter
    # transforms.ColorJitter(brightness=0.1),
    # transforms.ColorJitter(contrast=0.1),
    # transforms.ColorJitter(saturation=0.1),
    # transforms.ColorJitter(hue=0.3),

    # AddPepperNoise(0.93, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

path_img=os.path.join( '../data/shapenet/ShapeNetRendering/02691156/10155655850468db78d106ce0a280f87/rendering/00.png')
img = Image.open(path_img).convert('RGB')  # 0~255
img=transforms.Resize((128, 128))(img)
img_tensor = train_transform(img)



# ## 展示单张图片
# # 这里把转换后的 tensor 再转换为图片
# convert_img=transform_invert(img_tensor, train_transform)
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(convert_img)
# plt.show()
# plt.pause(0.5)
# plt.close()


# # 展示 FiveCrop 和 TenCrop 的图片
# ncrops, c, h, w = img_tensor.shape
# columns=2 # 两列
# rows= math.ceil(ncrops/2) # 计算多少行
# # 把每个 tensor ([c,h,w]) 转换为 image
# for i in range(ncrops):
#     img = transform_invert(img_tensor[i], train_transform)
#     plt.subplot(rows, columns, i+1)
#     plt.imshow(img)
# plt.show()
