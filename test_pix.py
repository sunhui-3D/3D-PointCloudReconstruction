from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(r"./loss")
sys.path.append(r"./models/")
sys.path.append(r"./utils/")
from datasets_FCP import GetPix3dDataset
import torch.backends.cudnn as cudnn
import argparse
import random
import torch
from os.path import join
import json
from repvgg_edge_nose_NEW_cmlp import generator
from torch.autograd import Variable
from icp import icp
import numpy as np
from average_meter import AverageMeter
from metrics import Metrics
from loss import Loss

def test_net(gen, testdataloader):
    # Testing loop
    test_losses = AverageMeter(['chamfer_loss', 'emd_loss_1', 'CD_loss_L1', 'CD_loss_L2', 'emd_loss_2',
                                'point_loss'])  # , 'uniform_loss', 'repulsion_loss'
    test_metrics = AverageMeter(Metrics.names())

    loss_fn = Loss().cuda()

    with torch.no_grad():

        data_iter = iter(testdataloader)
        n_samples = len(testdataloader)
        print('n_samples', n_samples)
        i = 0
        try:
            while i < len(testdataloader):
                data = data_iter.next()
                i += 1

                if i >= len(testdataloader):
                    break

                images, points_1024 = data

                images = Variable(images.float())
                points = Variable(points_1024.float())

                images = images.cuda()
                points = points.cuda()

                _, _, fake = gen(images)
                fake = fake.transpose(2, 1)  # b x n x c

                fake_icp = fake.cpu().detach().numpy()
                points_np = points
                points_np = points_np.cpu().detach().numpy()
                _pr_icp = []

                for index in range(fake_icp.shape[0]):
                    T, _, _ = icp(points_np[index], fake_icp[index], tolerance=1e-10, max_iterations=1024)
                    _pr_icp.append(np.matmul(fake_icp[index], T[:3, :3]) - T[:3, 3])

                fake_pr_icp = np.array(_pr_icp).astype('float32')
                fake_torch = torch.tensor(fake_pr_icp).cuda()

                chamfer_loss = loss_fn.get_chamfer_loss(fake_torch, points)
                emd_loss_1 = loss_fn.get_emd_loss(fake_torch, points)

                test_losses.update(
                    [chamfer_loss.item() * 100, emd_loss_1 * 100]) # , uniform_loss.item()*1000, repulsion_loss.item()*100000
                _metrics = Metrics.get(fake_torch, points)
                test_metrics.update(_metrics)

        except StopIteration:
            print("exception")


    return Metrics('ChamferDistance', test_metrics.avg()), Metrics('EMD_distance', test_metrics.avg())

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--cats', default='chair', type=str,
                    help='Category to train on : ["chair","sofa","table"]')
    parser.add_argument('--num_points', type=int, default=1024, help='number of pointcloud')
    parser.add_argument('--model', type=str, default='./output/repvgg_edge_nose_NEW_cmlp/%s/checkpoints/',  help='generator model path')

    opt = parser.parse_args()
    print (opt)

    blue = lambda x:'\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    cats = ['sofa', 'table', 'chair']
    print(cats)
    for cat in cats:

        print(cat)
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

        path =(opt.model)%c + 'model_best.pth.tar'

        gen = generator(num_points=opt.num_points)

        with open(path, "rb") as f:
            checkpoint = torch.load(f)
            gen.load_state_dict(checkpoint['state_dict'])#
        gen.cuda()
        gen.eval()

        test_dataset = GetPix3dDataset(data_dir, pix3d_models_dict, cat, opt.num_points)
        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

        chamfer_loss, emd_loss_1 = test_net(gen,  testdataloader)
        print(chamfer_loss, emd_loss_1)