import torch
from geomloss import SamplesLoss
import torch
import torch.nn as nn
import sys
sys.path.append(r"../metric/chamfer3D")
from dist_chamfer_3D import chamfer_3DDist

def cd(fake, points):

    cham3D = chamfer_3DDist()
    dist1, dist2, idx1, idx2= cham3D(fake, points)
    return torch.mean(dist1)+torch.mean(dist2)



# class ChamferLoss(nn.Module):

#     def __init__(self, ignore_zeros=False):
#         super(ChamferLoss, self).__init__()
#         self.use_cuda = torch.cuda.is_available()
#         self.ignore_zeros = ignore_zeros
#     def forward(self, preds, gts):
#         P = self.batch_pairwise_dist(gts, preds)
#         mins, _ = torch.min(P, 1)
#         loss_1 = torch.sum(mins)
#         mins, _ = torch.min(P, 2)
#         loss_2 = torch.sum(mins)
#         return loss_1 + loss_2

#     def batch_pairwise_dist(self, x, y):
#         bs, num_points_x, points_dim = x.shape
#         _, num_points_y, _ = y.shape
#         xx = torch.bmm(x, x.transpose(2, 1))
#         yy = torch.bmm(y, y.transpose(2, 1))
#         zz = torch.bmm(x, y.transpose(2, 1))
#         if self.use_cuda:
#             dtype = torch.cuda.LongTensor
#         else:
#             dtype = torch.LongTensor
#         diag_ind_x = torch.arange(0, num_points_x).type(dtype)
#         diag_ind_y = torch.arange(0, num_points_y).type(dtype)
#         rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
#             zz.transpose(2, 1))
#         ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
#         P = rx.transpose(2, 1) + ry - 2 * zz
#         return P



# def batch_pairwise_dist(x, y):
#     # 32, 2500, 3
#     bs, num_points_x, points_dim = x.size()
#     _, num_points_y, _ = y.size()
#     # print(bs, num_points, points_dim)
#     xx = torch.bmm(x, x.transpose(2, 1))
#     yy = torch.bmm(y, y.transpose(2, 1))
#     zz = torch.bmm(x, y.transpose(2, 1))
#     diag_ind_x = torch.arange(0, num_points_x).type(torch.cuda.LongTensor)
#     diag_ind_y = torch.arange(0, num_points_y).type(torch.cuda.LongTensor)
#     rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
#     ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
#     P = (rx.transpose(2, 1) + ry - 2 * zz)
#     return P

def batched_pairwise_dist(a, b):
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P

def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    P = batched_pairwise_dist(a, b)
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

def batch_NN_loss(x, y):
    # bs, num_points, points_dim = x.size()
    # dist1 = torch.sqrt(batch_pairwise_dist(x, y))
    # values1, indices1 = dist1.min(dim=2)
    #
    # dist2 = torch.sqrt(batch_pairwise_dist(y, x))
    # values2, indices2 = dist2.min(dim=2)
    # a = torch.div(torch.sum(values1,1), num_points)
    # b = torch.div(torch.sum(values2,1), num_points)
    # sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)
    # return sum

    P = batched_pairwise_dist(x, y)
    mins1, _ = torch.min(P, 1)
    mins2, _ = torch.min(P, 2)

    return torch.mean(mins1) + torch.mean(mins2), mins1, mins2

def batch_EMD_loss(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    batch_EMD = 0
    L = SamplesLoss(loss = 'gaussian', p=2, blur=.00005)
    for i in range(bs):
        loss = L(x[i], y[i])
        batch_EMD += loss
    emd = batch_EMD/bs
    return emd

def fscore(X, Y, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    _, dist1, dist2 = batch_NN_loss(X, Y)
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    fscore = torch.mean(fscore)
    precision_1 = torch.mean(precision_1)
    precision_2 = torch.mean(precision_2)
    fscore = fscore.requires_grad_()
    return fscore, precision_1, precision_2

if __name__ == '__main__':
    images_1 = torch.rand(32, 1024, 3).cuda()
    images_2 = torch.rand(32, 1024, 3).cuda()
    # cd, dist1, dist2 = batch_NN_loss(images_1, images_2)
    # pt_feature, precision_1, precision_2 = fscore(dist1, dist2)   
    cd = cd(images_1, images_2) 
    print(cd)
    # print(torch.mean(pt_feature))
    # print(torch.mean(precision_1))
    # print(torch.mean(precision_2))
