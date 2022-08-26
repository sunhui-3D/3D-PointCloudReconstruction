import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable


class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)
            for name, module in self.model.named_children():
                if name == 'classifier':
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
            classes = F.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19

from gradcam import GradCam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Grad-CAM')
    parser.add_argument('--image_name', default='both.png', type=str, help='the tested image name')
    parser.add_argument('--save_name', default='grad_cam.png', type=str, help='saved image name')

    opt = parser.parse_args()

    IMAGE_NAME = opt.image_name
    SAVE_NAME = opt.save_name
    test_image = (transforms.ToTensor()(Image.open(IMAGE_NAME))).unsqueeze(dim=0)
    model = vgg19(pretrained=True)
    if torch.cuda.is_available():
        test_image = test_image.cuda()
        model.cuda()
    grad_cam = GradCam(model)
    feature_image = grad_cam(test_image).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save(SAVE_NAME)
