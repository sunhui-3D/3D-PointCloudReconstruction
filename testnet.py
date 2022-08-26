from __future__ import print_function
import argparse
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(r"../loss")
sys.path.append(r"../metric/")
sys.path.append(r"../metric/emd/")
sys.path.append(r"../metric/chamfer3D/")
sys.path.append(r"../models/")
sys.path.append(r"../utils/")
import random
import torch
from os.path import join
import json
from datasets_FCP import GetShapenetDataset
import torch.backends.cudnn as cudnn
from repvgg_edge_nose_NEW_cmlp import generator
from torch.autograd import Variable
from icp import icp
import numpy as np
from datetime import datetime as dt
from tensorboardX import SummaryWriter
from average_meter import AverageMeter
from metrics import Metrics


def test_net(gen, testdataloader, cat, logger):
    # Testing loop
    test_losses = AverageMeter(['chamfer_loss', 'emd_loss_1'])
    test_metrics = AverageMeter(Metrics.names())

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

                images, points = data

                images = Variable(images.float())
                points = Variable(points.float())

                images = images.cuda()
                points = points.cuda()

                _, _, fake = gen(images)

                fake_icp = fake.cpu().detach().numpy()
                points_np = points
                points_np = points_np.cpu().detach().numpy()
                _pr_icp = []

                for index in range(fake_icp.shape[0]):
                    T, _, _ = icp(points_np[index], fake_icp[index], tolerance=1e-10, max_iterations=1024)
                    _pr_icp.append(np.matmul(fake_icp[index], T[:3, :3]) - T[:3, 3])

                fake_pr_icp = np.array(_pr_icp).astype('float32')
                fake_torch = torch.tensor(fake_pr_icp).cuda()

                _metrics = Metrics.get(fake_torch, points)
                test_metrics.update(_metrics)

                category = str(cat)
                logger.info('Test[%d/%d] Taxonomy = %s Metrics = %s' %
                         (i, n_samples, category, ['%.4f' % m for m in _metrics])),

        except StopIteration:
            print("exception")

    return Metrics('ChamferDistance', test_metrics.avg()), Metrics('EMD_distance', test_metrics.avg())

def test_main(gen,  testdataloader, cat, test_writer, epoch_idx, logger):
    chamfer_loss, emd_loss_1= test_net(gen,  testdataloader, cat, test_writer, epoch_idx, logger)
    return chamfer_loss, emd_loss_1

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    torch.backends.cudnn.benchmark=True

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--num_points', type=int, default=1024, help='umber of pointcloud, [1024, 2048]')
    parser.add_argument('--model', type=str, default='./output/repvgg_edge_nose_NEW_cmlp/%s/checkpoints/',  help='generator model path')

    opt = parser.parse_args()
    print (opt)

    blue = lambda x:'\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    gen = generator()#num_points=opt.num_points
    
    data_dir_imgs = '../data/shapenet/ShapeNetRendering/'
    data_dir_pcl = '../data/shapenet/ShapeNet_pointclouds/'

    cats = ['02691156','02828884', '02933112', '02958343','03636649','03211117','04090263','03001627','04530566','04379243', '03691459', '04401088',  '04256520']

    print(cats)
    for cat in cats:
        cat = [cat]
        print(cat)
        c = ''.join(cat)

        with open(join('../data/splits/', 'train_models.json'), 'r') as f:
            train_models_dict = json.load(f)

        with open(join('../data/splits/', 'val_models.json'), 'r') as f:
            val_models_dict = json.load(f)

        path =(opt.model)%c + 'model_best.pth.tar'

        with open(path, "rb") as f:
            checkpoint = torch.load(f)
            gen.load_state_dict(checkpoint['state_dict'])#
        gen.cuda()
        gen.eval()

        test_dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, val_models_dict, cat, opt.num_points)
        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                                  shuffle=False, num_workers=int(opt.workers))

        output_dir = os.path.join('./output/repvgg_edge_nose_NEW_cmlp/', '%s', '%s')
        log_dir = os.path.join(output_dir % (c, 'logs'), dt.now().isoformat())
        val_writer = SummaryWriter('%s/test'%log_dir)
        epoch_idx = 1

        try:
            os.makedirs(log_dir)
        except OSError:
            pass

        from logger import get_logger
        logger = get_logger((opt.model)%c + 'logging_test.log')

        chamfer_loss, emd_loss_1 = test_main(gen,  testdataloader, cat, val_writer, epoch_idx, logger)
