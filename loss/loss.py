import os,sys
sys.path.append(os.getcwd())
sys.path.append(r"../metric/emd/")
sys.path.append(r"../metric/chamfer3D/")
sys.path.append(r"../models/")
sys.path.append(r"../utils/")
sys.path.append('../')
from dist_chamfer_3D import chamfer_3DDist
import emd_module as emd_func
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self,radius=1.0):
        super(Loss,self).__init__()
        self.radius=radius
        # self.knn_uniform=KNN(k=2,transpose_mode=True)
        # self.knn_repulsion=KNN(k=20,transpose_mode=True)
    def get_emd_loss(self,pred,gt,radius=1.0):
        '''
        pred and gt is B N 3
        '''
        emd = emd_func.emdModule().cuda()
        emd_1, _ = emd(pred, gt, eps=0.05, iters=3000)#50  #0.005
        # emd_loss = np.sqrt(emd_1.cpu()).mean()
        emd_loss = torch.sqrt(emd_1).mean(1).mean()
        # emd_loss_np = emd_loss.item()#*100 #already mul 1000

        return emd_loss

    def get_chamfer_loss(self, pred, gt):
        '''
        pred and gt is B N 3
        '''
        chamLoss = chamfer_3DDist().cuda()
        dist1, dist2, idx1, idx2 = chamLoss(pred, gt)
        chamfer_loss = torch.mean(dist1)+torch.mean(dist2)#0.55*
        return chamfer_loss

if __name__ == "__main__":
    import torch
    point = torch.rand(4, 1024, 3).cuda()
    pre_point = torch.rand(4, 1024, 3).cuda()

    loss=Loss().cuda()
    chamfer_loss=loss.get_chamfer_loss(pre_point, point)
    print('chamfer_loss', chamfer_loss)
    emd_loss_1=loss.get_emd_loss(pre_point, point)
    print('emd_loss_1', emd_loss_1)

