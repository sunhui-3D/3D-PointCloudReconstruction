from __future__ import print_function
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
sys.path.append(os.getcwd())
sys.path.append('..')
sys.path.append(r"./loss")
sys.path.append(r"./metric/emd/")
sys.path.append(r"./metric/chamfer3D/")
sys.path.append(r"./models/")
sys.path.append(r"./utils/")
import random
import torch
from os.path import join
import json
from datasets_old import GetShapenetDataset
import torch.backends.cudnn as cudnn
from repvgg_edge_nose_NEW_cmlp import generator
import torch.optim as optim
from torch.autograd import Variable
from loss import Loss
from logger import get_logger
import time
from tensorboardX import SummaryWriter
from datetime import datetime as dt
from average_meter import AverageMeter
import shutil
import torchvision.transforms as transforms

def train_net(cat):

    torch.set_printoptions(profile='full')
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='', help='category')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#注意windows和ubuntu的区别
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')    
    parser.add_argument('--lr', type=float, default = '0.0005', help='learning rate')
    parser.add_argument('--resume', type=bool, default=False,  help='pretrained checkpoint')
    parser.add_argument('--lambda_cd', type=int, default='100',  help='chamfer_loss weight')
    parser.add_argument('--lambda_emd', type=int, default='100',  help='emd_loss weight')
    parser.add_argument('--train_save_freq', type=int, default='20',  help='train weoght save frequence')
    parser.add_argument('--num_points', type=int, default=1024, help='number of epochs to train for, [1024, 2048]')
    parser.add_argument('--dir_path', type=str, default='./output/repvgg_edge_nose_NEW_cmlp/',  help='output folder')
    parser.add_argument('--splits_path', type=str, default='/home/chenwenyu/my_all_model/3D-MSNet/data/splits/',  help='splits_path')
    parser.add_argument('--data_dir_imgs', type=str, default='/home/chenwenyu/my_all_model/3D-MSNet/data/shapenet/ShapeNetRendering/',  help='data_dir_imgs')
    parser.add_argument('--data_dir_pcl', type=str, default='/home/chenwenyu/my_all_model/3D-MSNet/data/shapenet/ShapeNet_pointclouds/',  help='data_dir_pcl')

    opt = parser.parse_args()
    print (opt)

    blue = lambda x:'\033[94m' + x + '\033[0m'
    
    opt.category = cat
    print('opt.category', opt.category)
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)

    with open(join(opt.splits_path, 'train_models.json'), 'r') as f:
        train_models_dict = json.load(f)
    with open(join(opt.splits_path, 'val_models.json'), 'r') as f:
        val_models_dict = json.load(f)

    train_transform = transforms.Compose([

        transforms.ToPILImage(),
        transforms.RandomChoice([transforms.RandomVerticalFlip(p=0.8), transforms.RandomHorizontalFlip(p=0.8)]),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    dataset = GetShapenetDataset(opt.data_dir_imgs, opt.data_dir_pcl, train_models_dict, opt.category, opt.num_points, variety=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers), drop_last=True,pin_memory=True)

# Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    cudnn.benchmark = True

# Set up folders for logs and checkpoints  
    category = ''.join(opt.category)
    output_dir = os.path.join(opt.dir_path, '%s', '%s')
    log_dir = os.path.join(output_dir % (category, 'logs'), dt.now().isoformat())
    log_dir_train = (os.path.join(log_dir, '%s')) % 'train'
    log_dir_test = (os.path.join(log_dir, '%s')) % 'test'
    ckpt_dir = output_dir % (category, 'checkpoints')
    try:
        os.makedirs(ckpt_dir)
    except OSError:
        pass

#get_logger
    logger = get_logger(ckpt_dir + '/' + 'logging.log')

# Create the networks
    gen = generator()
    gen.cuda()

# Create tensorboard writers
    train_writer = SummaryWriter('%s/train'%log_dir)
    val_writer = SummaryWriter('%s/test'%log_dir)

# init loss function
    loss_fn = Loss().cuda()

# Create the optimizers
    optimizerG = optim.Adam(gen.parameters(), lr = opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)

#load pretrainde model if exists
    best_metrics = None
    best_chamfer_loss = None
    best_emd_loss = None
    all_epoch_time = 0.
    if opt.resume:
        print('pretrained model is true')
        checkpoint = torch.load(ckpt_dir +  '/' + "model_best.pth.tar")
        opt.start_epoch = checkpoint['epoch']
        best_chamfer_loss = checkpoint['best_chamfer_loss']
        best_emd_loss = checkpoint['best_emd_loss']
        gen.load_state_dict(checkpoint['state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        all_epoch_time = checkpoint['train_time']

# Training/Testing the network
    for epoch in range(opt.start_epoch+1, opt.nepoch+1):

        epoch_start_time = time.time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['chamfer_loss', 'emd_loss'])

        data_iter = iter(dataloader)
        n_batches = len(dataloader)
        i = 0

        gen.train()
        batch_end_time = time.time()

        while i < len(dataloader):

            data_time.update(time.time() - batch_end_time)

            data = data_iter.next()
            i += 1
    
            images, points, xangle, _= data
    
            points = Variable(points.float()).cuda()
            images = Variable(images.float()).cuda()

            _, _, fake = gen(images)

            if epoch >30 and epoch <= 50:
                chamfer_loss   = loss_fn.get_chamfer_loss(fake.transpose(2, 1), points)
                emd_loss       = loss_fn.get_emd_loss(fake.transpose(2, 1), points) 
                total_loss     = emd_loss*opt.lambda_emd
            elif epoch > 0 and epoch <=30: 
                chamfer_loss   = loss_fn.get_chamfer_loss(fake.transpose(2, 1), points)
                emd_loss       = loss_fn.get_emd_loss(fake.transpose(2, 1), points) 
                total_loss     = chamfer_loss*opt.lambda_cd + emd_loss*opt.lambda_emd
            else:
                continue

            losses.update([chamfer_loss.item()*100, emd_loss.item()*100])

            gen.zero_grad()
            total_loss.backward()
            optimizerG.step()
            # lr_scheduler.step()

            n_itr = (epoch - 1) * n_batches + i
            train_writer.add_scalar('scalar/total_loss', total_loss, n_itr)
            train_writer.add_scalar('scalar/chamfer_loss', chamfer_loss, n_itr)
            train_writer.add_scalar('scalar/emd_loss', emd_loss, n_itr)  

            batch_time.update(time.time() - batch_end_time)
            batch_end_time = time.time()
            logger.info('[Category %s] [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (category, epoch, opt.nepoch, i, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))
        
        if epoch % 10 == 0 and epoch != 0:
            if epoch >= 30 and epoch < 40:
                opt.lr = opt.lr * 0.01
            elif epoch>=40:
                opt.lr = opt.lr * 0.001
            else:
                opt.lr = opt.lr * 0.1
            for param_group in optimizerG.param_groups:
                param_group['lr'] = opt.lr

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        all_epoch_time = all_epoch_time + epoch_time
        train_writer.add_scalar('Loss/Epoch/chamfer_loss', losses.avg(0), epoch)
        train_writer.add_scalar('Loss/Epoch/emd_loss', losses.avg(1), epoch)
        logger.info(
                     '[[Category %s] Epoch %d/%d] EpochTime = %.3f (s) All_epoch_time = %.3f (s) Losses = %s' %
                     (category, epoch, opt.nepoch, epoch_time, all_epoch_time, ['%.4f' % l for l in losses.avg()]))

## validate the model
        if epoch==10 or epoch==30 or epoch==50:
            try:
                chamfer_loss, emd_loss = validate(gen,val_models_dict, opt, logger)

            # remember best and save checkpoint
                is_best = chamfer_loss.better_than(best_chamfer_loss) and emd_loss.better_than(best_emd_loss)
                print('is_best', is_best)

# Save checkpoints
                save_checkpoint({
                       'epoch': epoch,
                       "model_name": ckpt_dir,
                       'state_dict': gen.state_dict(),
                       'best_chamfer_loss': chamfer_loss,
                       'best_emd_loss': emd_loss,
                       'optimizerG': optimizerG.state_dict(),
                       'train_time':all_epoch_time,
                }, is_best, category, ckpt_dir,epoch)

            except OSError:  
                torch.save(gen.state_dict(), '%s/pth_%d_%s.pth' % (ckpt_dir, epoch, category))

    train_writer.close()  
    val_writer.close()  

def save_checkpoint(state, is_best, category, ckpt_dir,epoch):

    filename = ckpt_dir + '/' + "%s_checkpoint_%s.pth.tar"%(category, epoch)
    torch.save(state, filename)

    if is_best:
        message = ckpt_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(filename, message)

def validate(model, val_models_dict, opt, test_writer, epoch_idx, logger):
    val_transform = transforms.Compose(
        [
         transforms.ToTensor(),#numpy->tensor 
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),#numpy
         ])
    test_dataset = GetShapenetDataset(opt.data_dir_imgs, opt.data_dir_pcl, val_models_dict, opt.category, opt.num_points, val_transform)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=int(8), pin_memory=True, drop_last=False)
    model.eval()

    print(len(test_dataset))

    with torch.no_grad():
        from testnet import test_main
        chamfer_loss, emd_loss = test_main(model, testdataloader, opt.category, test_writer, epoch_idx, logger) 
        return chamfer_loss, emd_loss

def main():

    cats = ['02828884' ]
    #    {"airplane":02691156, "bench": '02828884' "cabinet":02933112, "car":02958343, "lamp":03636649,
    #    "monitor":03211117, "rifle":04090263, "sofa":04256520, "speaker":03691459, "table":04379243,
    #    "telephone":04401088, "vessel":04530566, "chair":03001627}
    print(cats)
    for cat in cats:
        cat = [str(cat)]
        start_category_train = time.time()
        train_net(cat)
        end_category_train = time.time()
        print('cat: %s  this category train time: %f h' %(cat, (end_category_train-start_category_train)/3600)) 

if __name__ == '__main__':

    start = time.time()
    main()
    end = time.time()
    print('all categories run time :%f h'%(end-start)/3600)
