from __future__ import print_function
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append(os.getcwd())
sys.path.append(r"../loss")
sys.path.append(r"../metric/emd/")
sys.path.append(r"../metric/chamfer3D/")
sys.path.append(r"../models/")
sys.path.append(r"../utils/")
import time
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import torch.utils.data as data
from os.path import join
import json
import torch
import numpy as np
import cv2
import random
from utils import distance_squre
import utils

NUM_VIEWS = 24
images = []
HEIGHT = 128
WIDTH = 128
PAD = 35

class GetShapenetDataset(data.Dataset):
    def __init__(self, data_dir_imgs, data_dir_pcl, models, cats, numpoints=1024, variety=False):
        self.data_dir_imgs = data_dir_imgs
        self.data_dir_pcl = data_dir_pcl
        self.models = models
        self.modelnames = []
        self.size = 0
        self.numpoints = numpoints
        self.variety = variety
        # self.transform = transform

        for cat in cats:
            for filename in self.models[cat]:
                for i in range(NUM_VIEWS):
                    self.size = self.size + 1
                    self.modelnames.append(filename)

    def __getitem__(self, index):

        pcl_path = self.data_dir_pcl + self.modelnames[index] + '/pointcloud_' + str(self.numpoints) + '.npy'
        pcl_gt = np.load(pcl_path)
        pcl_path_index = self.data_dir_pcl + self.modelnames[index]
        key1_path = pcl_path_index
        key1 = os.path.exists(key1_path + '/pointcloud_' + '128' + '.npy')
        key2 = os.path.exists(key1_path + '/pointcloud_' + '256' + '.npy')
        print(key1, key2, key1 and key2)

        if (key1 and key2) == False:

            points = torch.unsqueeze(torch.Tensor(pcl_gt), 0)

            points = Variable(points.float())

            batch_size = points.size()[0]
            real_center = torch.FloatTensor(batch_size, 1, 1024, 3)
            points = torch.unsqueeze(points, 1)

            choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                      torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
            for m in range(batch_size):
                index = random.sample(choice, 1)
                distance_list = []
                p_center = index[0]
                for n in range(1024):
                    distance_list.append(distance_squre(points[m, 0, n], p_center))
                distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

                for sp in range(1024):
                    real_center.data[m, 0, sp] = points[m, 0, distance_order[sp][0]]

            real_center = real_center.cuda()

            real_center = Variable(real_center, requires_grad=True)
            real_center = torch.squeeze(real_center, 1)
            real_center_key1_idx = utils.farthest_point_sample(real_center, 128, RAN=False)
            real_center_key1 = utils.index_points(real_center, real_center_key1_idx)

            real_center_key2_idx = utils.farthest_point_sample(real_center, 256, RAN=True)
            real_center_key2 = utils.index_points(real_center, real_center_key2_idx)

            np.save(key1_path + '/pointcloud_' + '128' + '.npy', real_center_key1.squeeze(0).detach().cpu().numpy())
            np.save(key1_path + '/pointcloud_' + '256' + '.npy', real_center_key2.squeeze(0).detach().cpu().numpy())

        return  pcl_gt

    def __len__(self):
        return self.size


class GetPix3dDataset(data.Dataset):
    def __init__(self, data_dir, models, cats, numpoints=1024, save=False):
        self.save = save
        self.data_dir = data_dir
        self.models = models
        self.size = 0
        self.cats = cats
        self.numpoints = numpoints
        self.imgpaths = []
        self.maskpaths = []
        self.modelpaths = []
        self.bbox = []
        pcl = 'pcl_' + str(self.numpoints)

        for model in self.models:
            if model['category'] == self.cats:
                # model/[category]/[modelname] /model.obj
                modelpath = model['model'].replace("model", pcl)  # pcl_1024/[category]/[modelname]/pcl_1024.obj
                modelpath = modelpath.replace("pcl_1024", "model", 1)  # model/[category]/[modelname]/pcl_1024.obj
                modelpath = modelpath.replace("obj", 'npy')  # model/[category]/[modelname]/pcl_1024.npy
                pcl_path = self.data_dir + 'pointclouds/' + modelpath
                if os.path.exists(pcl_path):
                    self.imgpaths.append(model['img'])
                    self.maskpaths.append(model['mask'])
                    self.modelpaths.append(model['model'])
                    self.bbox.append(model['bbox'])
                    self.size = self.size + 1

    def __getitem__(self, index):
        img_path = self.data_dir + self.imgpaths[index]
        mask_path = self.data_dir + self.maskpaths[index]
        pcl = 'pcl_' + str(self.numpoints)
        modelpath = self.modelpaths[index].replace("model", pcl)
        modelpath = modelpath.replace("pcl_1024", "model", 1)
        modelpath = modelpath.replace("obj", 'npy')
        pcl_path = self.data_dir + 'pointclouds/' + modelpath
        img_name = img_path[-8:-4]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path)
        if not image.shape[0] == mask_image.shape[0] or not image.shape[1] == mask_image.shape[1]:
            mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
        image = image * mask_image
        image = image[self.bbox[index][1]:self.bbox[index][3], self.bbox[index][0]:self.bbox[index][2], :]
        current_size = image.shape[:2]
        ratio = float(HEIGHT - PAD) / max(current_size)
        new_size = tuple([int(x * ratio) for x in current_size])
        image = cv2.resize(image, (new_size[1], new_size[0]))  # new_size should be in (width, height) format
        delta_w = WIDTH - new_size[1]
        delta_h = HEIGHT - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        image = np.transpose(image, (2, 0, 1))

        xangle = np.pi / 180. * -90
        yangle = np.pi / 180. * -90
        pcl_gt = rotate(rotate(np.load(pcl_path), xangle, yangle), xangle)
        if not self.save:
            return image, pcl_gt
        else:
            return image, pcl_gt, img_name

    def __len__(self):
        return self.size


def rotate(xyz, xangle=0, yangle=0, zangle=0):
    rotmat = np.eye(3)
    rotmat = rotmat.dot(np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(xangle), -np.sin(xangle)],
        [0.0, np.sin(xangle), np.cos(xangle)],
    ]))
    rotmat = rotmat.dot(np.array([
        [np.cos(yangle), 0.0, -np.sin(yangle)],
        [0.0, 1.0, 0.0],
        [np.sin(yangle), 0.0, np.cos(yangle)],
    ]))
    rotmat = rotmat.dot(np.array([
        [np.cos(zangle), -np.sin(zangle), 0.0],
        [np.sin(zangle), np.cos(zangle), 0.0],
        [0.0, 0.0, 1.0]
    ]))

    return xyz.dot(rotmat)


def train_net(cat):
    torch.set_printoptions(profile='full')

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='', help='category')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--num_points', type=int, default=1024, help='number of epochs to train for, [1024, 2048]')
    parser.add_argument('--dir_path', type=str,default='../output/repvgg_edge_nose_NEW_cmlp_multi/',help='output folder')
    parser.add_argument('--splits_path', type=str, default='../data/splits/', help='splits_path')
    parser.add_argument('--data_dir_imgs', type=str,default='../data/shapenet/ShapeNetRendering/', help='data_dir_imgs')
    parser.add_argument('--data_dir_pcl', type=str,default='../data/shapenet/ShapeNet_pointclouds/', help='data_dir_pcl')
    parser.add_argument('--data_pcl_output', type=str,default='../data/shapenet/ShapeNet_pointclouds/%s/', help='data_dir_pcl')
    parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    opt.category = cat
    print('opt.category', opt.category)
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    with open(join(opt.splits_path, 'train_models.json'), 'r') as f:
        train_models_dict = json.load(f)
    with open(join(opt.splits_path, 'val_models.json'), 'r') as f:
        val_models_dict = json.load(f)
    dataset = GetShapenetDataset(opt.data_dir_imgs, opt.data_dir_pcl, train_models_dict, opt.category, opt.num_points,
                                 variety=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers), drop_last=True,
                                             pin_memory=True)

    test_dataset = GetShapenetDataset(opt.data_dir_imgs, opt.data_dir_pcl, val_models_dict, opt.category, opt.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=int(0), pin_memory=True, drop_last=False)

    # data_iter = iter(dataloader)
    data_iter_1 = iter(testdataloader)
    i = 0

    while i < len(dataloader):
        # data = data_iter.next()
        data_1 = data_iter_1.next()
        i += 1
        # points = data


def main():
    cats = ['02958343','03636649','04090263','03001627','04530566','04379243', '04256520','02691156','03211117', '02933112','03691459','04401088','02828884']
    for cat in cats:
        cat = [str(cat)]
        start_category_train = time.time()
        print(cat)
        train_net(cat)
        end_category_train = time.time()
        print('cat: %s  this category train time: %f h' % (cat, (end_category_train - start_category_train) / 3600))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('all categories run time :%s h' % (end - start) / 3600)


