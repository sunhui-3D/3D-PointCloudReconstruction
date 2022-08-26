from __future__ import print_function
import sys
sys.path.append(r"../")
import argparse
from os.path import join
import json
import numpy as np
from datasets_FCP import GetPix3dDataset
import torch
import torch.backends.cudnn as cudnn
from repvgg_edge_nose_NEW_cmlp import generator
from torch.autograd import Variable
import matplotlib.pylab as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM_VIEWS = 1
cudnn.benchmark = True
azim = -45
elev = -165
scale = 0.45
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')  #save img batchsize = 1
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--cats', default='chair', type=str,
                    help='Category to train on : ["chair","sofa","table"]')
parser.add_argument('--num_points', type=int, default=1024, help='number of pointcloud')
parser.add_argument('--model', type=str, default='./output/repvgg_edge_nose_NEW_cmlp/%s/checkpoints/',  help='generator model path')
opt = parser.parse_args()

cats = [ 'sofa','table']

for cat in cats:

    with open(join('../data/pix3d/', 'pix3d.json'), 'r') as f:
        pix3d_models_dict = json.load(f)

    data_dir = '../data/pix3d/'

    if cat == 'chair':
        c = '03001627'
    elif cat == 'sofa':
        c = '04256520'
    else:
        c = '04379243'

    c = ''.join(c)

    save_path = '/home/chenwenyu/3D-RAGNet/output/repvgg_edge_nose_NEW_cmlp/' + c + '/pix3d_img_new/'

    # save_path = './pix3d_img/' + opt.cats + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # gen = generator(num_points=opt.num_points)
    # gen.cuda().eval()
    # with open(opt.model, "rb") as f:
    #     gen.load_state_dict(torch.load(f))

    pickle_path = '/home/chenwenyu/3D-RAGNet/output/' + 'repvgg_edge_nose_NEW_cmlp/' + c + '/checkpoints/model_best.pth.tar'#02691156_checkpoint_30.pth.tar'

    gen = generator()
    gen.cuda().eval()
    with open(pickle_path, "rb") as f:
        checkpoint = torch.load(f)
        gen.load_state_dict(checkpoint['state_dict'])

    # with open(join('data/splits/', 'pix3d.json'), 'r') as f:
    #     pix3d_models_dict = json.load(f)
    # data_dir = './data/pix3d/'
    test_dataset = GetPix3dDataset(data_dir, pix3d_models_dict, cat, 1024, save=True)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
    print(len(test_dataset))

    with torch.no_grad():
        data_iter = iter(testdataloader)
        index = 0
        while index < len(testdataloader):
            data = data_iter.next()
            index += 1

            if index >= len(testdataloader):
                break

            images, points, img_name = data

            if os.path.exists(save_path + img_name[0] + '_gt.png') == False:

                images = Variable(images.float())
                points = Variable(points.float())

                images = images.cuda()
                points = points.cuda()


                _, _,fake = gen(images)
                fake = fake.transpose(2, 1)  # b x n x c

                fake = np.squeeze(fake.cpu().detach().numpy())  # n x c
                points = np.squeeze(points.cpu().detach().numpy())  # n x c

                # save groundtruth img
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_xlim(-scale, scale)
                ax.set_ylim(-scale, scale)
                ax.set_zlim(-scale, scale)
                for i in range(len(points)):
                    ax.scatter(points[i, 1], points[i, 2], points[i, 0], c='r', s=5, depthshade=True)
                ax.axis('off')
                ax.view_init(azim=azim, elev=elev)
                plt.savefig(save_path + img_name[0] + '_gt.png')
                # plt.show()
                plt.close()

                # save predict img
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_xlim(-scale, scale)
                ax.set_ylim(-scale, scale)
                ax.set_zlim(-scale, scale)
                for i in range(len(fake)):
                    ax.scatter(fake[i, 1], fake[i, 2], fake[i, 0], c='r', s=5, depthshade=True)
                ax.axis('off')
                ax.view_init(azim=azim, elev=elev)
                plt.savefig(save_path + img_name[0] + '_pr.png')
                # plt.show()
                plt.close()

                if index % 5 == 0:
                    print("saving " + str(index) + " imgs")
            else:
                print('exists')