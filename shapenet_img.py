from __future__ import print_function
import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(r"./metric/emd/")
sys.path.append(r"./metric/chamfer3D/")
sys.path.append(r"./models/")
sys.path.append(r"./utils/")
import torch.backends.cudnn as cudnn
from repvgg_edge_nose_NEW_cmlp import generator
from torch.autograd import Variable
import cv2
import matplotlib.pylab as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
NUM_VIEWS = 1
cudnn.benchmark = True
azim = -50
elev = -145
scale = 0.4
base_path = '../data/shapenet/'


cats = ['04090263']
for category in cats:
    save_path = './output/repvgg_edge_nose_NEW_cmlp/' + category + '/shapenet_img/'
    pickle_path = './output/' + 'repvgg_edge_nose_NEW_cmlp/' + category + '/checkpoints/model_best.pth.tar'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    file_paths = os.listdir(base_path + 'ShapeNetRendering/' + category + '/')

    gen = generator()
    gen.cuda().eval()
    with open(pickle_path, "rb") as f:
        checkpoint = torch.load(f)
        gen.load_state_dict(checkpoint['state_dict'])


    for file_path in file_paths:
        point_path = base_path + 'ShapeNet_pointclouds/' + category + '/' + file_path + '/pointcloud_1024.npy'
        if os.path.exists(save_path + file_path + '_00_pr_128.png') == False:
            for i in range(NUM_VIEWS):
                img_path = base_path + 'ShapeNetRendering/' + category + '/' + file_path + '/rendering/' + (str(int(i % NUM_VIEWS)).zfill(2) + '.png')
                image = cv2.imread(img_path)[4:-5, 4:-5, :3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, (2, 0, 1))
                image = torch.Tensor(image)
                image = image.unsqueeze(0)

                image = Variable(image.float())
                image = image.cuda()

                points_128, point_256, _ = gen(image)
                points_128 = points_128.cpu().detach().numpy()
                points_128 = np.squeeze(points_128)
                points_128 = np.transpose(points_128, (1, 0))

                point_256 = point_256.cpu().detach().numpy()
                point_256 = np.squeeze(point_256)
                point_256 = np.transpose(point_256, (1, 0))

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_xlim(-scale, scale)
                ax.set_ylim(-scale, scale)
                ax.set_zlim(-scale, scale)
                for i in range(len(points_128)):
                    ax.scatter(points_128[i, 1], points_128[i, 2], points_128[i, 0], c='r', s=5, depthshade=True)#'#00008B'1f77d4
                ax.axis('off')
                ax.view_init(azim=azim, elev=elev)
                plt.savefig(save_path + file_path + '_' + (str(int(i % NUM_VIEWS)).zfill(2) + '_pr_128.png'))
                plt.close()

                # save predict img
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_xlim(-scale, scale)
                ax.set_ylim(-scale, scale)
                ax.set_zlim(-scale, scale)
                for i in range(len(point_256)):
                    ax.scatter(point_256[i, 1], point_256[i, 2], point_256[i, 0], c='r', s=5, depthshade=True)#'#00008B'1f77d4
                ax.axis('off')
                ax.view_init(azim=azim, elev=elev)
                plt.savefig(save_path + file_path + '_' + (str(int(i % NUM_VIEWS)).zfill(2) + '_pr_256.png'))
                plt.close()

            print("saving model img...")