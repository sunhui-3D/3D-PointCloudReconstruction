import torch
import numpy as np
import time
from emd_module import emdModule

def test_emd():
    x1 = torch.rand(20, 2048, 3).cuda()
    x2 = torch.rand(20, 2048, 3).cuda()
    emd = emdModule()
    start_time = time.perf_counter()
    dis, assigment = emd(x1, x2, 0.05, 3000)
    dis1, assigment1 = emd(x1, x2, 0.05, 50)
    print(np.sqrt(dis1.cpu()).mean())

    mean1 = torch.sqrt(dis).mean(1)
    mean0 = torch.sqrt(dis).mean(0)
    print("dis", dis, dis.shape)  # torch.Size([bs, num_points])
    print("mean1", mean1, mean1.shape)  # torch.Size([bs])
    print("mean0", mean0, mean0.shape)  # torch.Size([num_points]) -- wrong
    print("Input_size: ", x1.shape)
    print("Runtime: %lfs" % (time.perf_counter() - start_time))
    print("EMD: %lf" % np.sqrt(dis.cpu()).mean())
    print("|set(assignment)|: %d" % assigment.unique().numel())
    assigment = assigment.cpu().numpy()
    assigment = np.expand_dims(assigment, -1)
    x2 = np.take_along_axis(x2, assigment, axis=1)
    d = (x1 - x2) * (x1 - x2)
    print("Verified EMD: %lf" % np.sqrt(d.cpu().sum(-1)).mean())

if __name__ == '__main__':
    test_emd()