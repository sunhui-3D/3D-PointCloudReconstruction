import torch
import numpy as np

def cont_proj(pcl, grid_h, grid_w,device ,sigma_sq=0.5):
    '''
    Continuous approximation of Orthographic projection of point cloud
    to obtain Silhouette
    args:
            pcl: float, (N_batch,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W);
                      output silhouette
    '''
    # print('pcl.shape', pcl.shape)
    xyz=torch.split(pcl,1,dim=2)
    # print(type(xyz))
    x=(xyz[0]+1)*grid_h/2
    y= (xyz[1]+1)*grid_w/2
    z = xyz[2]

    # x = xyz[0]*grid_h
    # y = xyz[1]*grid_w
    # z = xyz[2]

    # pcl_norm = torch.cat([x, y, z], 2)
    pcl_xy = torch.cat([x,y], 2).to(device)
    # print('pcl_xy', pcl_xy.shape)
    #
    # pcl_xy = pcl_xy.squeeze().cpu().numpy()
    # print('pcl_xy', pcl_xy.shape)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # # 设置标题
    # ax1.set_title('Scatter Plot')
    # # 设置X轴标签
    # plt.xlabel('X')
    # # 设置Y轴标签
    # plt.ylabel('Y')
    # # 画散点图
    # ax1.scatter(x=pcl_xy[:, 0], y=pcl_xy[:, 1], c='r', marker='o')
    # # 设置图标
    # plt.legend('x1')
    # # plt.imshow(images)
    # plt.savefig('%s_images_.png' % i)  # 图像保存
    # # 显示所画的图
    # plt.show()

    out_grid = torch.meshgrid(torch.range(0,grid_h-1), torch.range(0,grid_w-1))#设置一个网格[0~63,0~63]
    out_grid = [out_grid[0].float(), out_grid[1].float()]#设置网络的数据类型
    grid_z = torch.unsqueeze(torch.zeros_like(out_grid[0]), 2) # (H,W,1)#
    grid_xyz = torch.cat([torch.stack(out_grid,2), grid_z],2)  # (H,W,3)
    grid_xy = torch.stack(out_grid,2).to(device)  #堆叠结果64*64*2 (H,W,2)
    # print(grid_xy)
    # print(torch.unsqueeze(pcl_xy, 2).shape)
    # print(torch.unsqueeze(torch.unsqueeze(pcl_xy, 2), 2))
    # print(torch.unsqueeze(torch.unsqueeze(pcl_xy, 2), 2))
    grid_diff = torch.unsqueeze(torch.unsqueeze(pcl_xy, 2), 2).to(device) - grid_xy # (BS,N_PTS,H,W,2)（BS，1024,1,1,2）-（64,64,2）=（BS，1024,64,64,2）
    grid_val = apply_kernel(grid_diff, sigma_sq)    # (BS,N_PTS,H,W,2)（BS，1024,64,64,2）
    grid_val = grid_val[:,:,:,:,0]*grid_val[:,:,:,:,1]  # (BS,N_PTS,H,W)（BS，1024,64,64）
    grid_val = torch.sum(grid_val,1)   # (BS,H,W)
    # th=torch.nn.Tanh()
    # grid_val = th(grid_val)
    return grid_val

def disc_proj(pcl, grid_h, grid_w):
    '''
    Discrete Orthographic projection of point cloud
    to obtain Silhouette
    Handles only batch size 1 for now
    args:
            pcl: float, (N_batch,N_Pts,3); input point cloud
                     values assumed to be in (-1,1)
            grid_h, grid_w: int, ();
                     output depth map height and width
    returns:
            grid_val: float, (N_batch,H,W); output silhouette
    '''
    xyz = torch.split(pcl, 1, dim=2)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]


    pcl_norm = torch.cat([x, y, z], 2)
    pcl_xy = torch.cat([x,y], 2)
    xy_indices = pcl_xy.long()
    #TODO
    xy_values = torch.ones_like(xy_indices[:,:,0]).float()
    out_grid = torch.zeros(grid_h,grid_w).scatter(0, xy_indices[0],xy_values)
    out_grid = torch.unsqueeze(out_grid,0)
    return out_grid

def apply_kernel(x, sigma_sq=0.5):
    '''
    Get the un-normalized gaussian kernel with point co-ordinates as mean and
    variance sigma_sq
    args:
            x: float, (BS,N_PTS,H,W,2); mean subtracted grid input
            sigma_sq: float, (); variance of gaussian kernel
    returns:
            out: float, (BS,N_PTS,H,W,2); gaussian kernel
    '''
    out = (torch.exp(-(x**2)/(2.*sigma_sq)))
    return out

def perspective_transform(xyz, device,batch_size):
    '''
    相机坐标系->图像坐标系->像素坐标系
    Perspective transform of pcl; Intrinsic camera parameters are assumed to be
    known (here, obtained using parameters of GT image renderer, i.e. Blender)
    Here, output grid size is assumed to be (64,64) in the K matrix
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
    returns:
            xyz_out: float, (BS,N_PTS,3); perspective transformed point cloud
    '''
    # K = np.array([
    #         [120., 0., -32.],
    #         [0., 120., -32.],
    #         [0., 0., 1.]]).astype(np.float32)
    K = np.array([
            [120., 0., -32.],
            [0., 120., -32.],
            [0., 0., 1.]]).astype(np.float32)
    K = np.expand_dims(K, 0)
    K = np.tile(K, [batch_size,1,1])
    # print('k', K.shape)

    xyz_out = torch.matmul(torch.from_numpy(K).float().to(device), xyz.permute(0,2,1))
    # print('xyz_o', xyz_out.shape)
    # print('xyz', xyz.shape)
    # print('abs(torch.unsqueeze(xyz[:,:,2],1))',abs(torch.unsqueeze(xyz[:,:,2],1)).shape )#1,1,1024
    # print(xyz_out[:,:2].shape)
    # print(xyz_out[:,:2].shape, xyz_out[:,:2])
    # print(abs(torch.unsqueeze(xyz[:,:,2],1)).shape)
    xy_out = xyz_out[:,:2]/abs(torch.unsqueeze(xyz[:,:,2],1))
    # print(xy_out.shape)
    xyz_out = torch.cat([xy_out, abs(xyz_out[:,2:])],1)
    # print('xyz_out', xyz_out.shape)
    # print('perspective_transform', xyz_out.permute(0,2,1))
    return xyz_out.permute(0,2,1)

def world2cam(xyz, az, el, batch_size,device, N_PTS=1024):
    '''
    世界坐标系->相机坐标系
    Convert pcl from world co-ordinates to camera co-ordinates
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            az: float, (BS); azimuthal angle of camera in radians
            elevation: float, (BS); elevation of camera in radians
            batch_size: int, (); batch size
            N_PTS: float, (); number of points in point cloud
    returns:
            xyz_out: float, (BS,N_PTS,3); output point cloud in camera
                        co-ordinates
    '''
    # Distance of object from camera - fixed to 2
    d = 2.5
    # d = 5
    # Calculate translation params
    # Camera origin calculation - az,el,d to 3D co-ord
    tx, ty, tz = [0, 0, d]
    rotmat_az =torch.stack(
        ((torch.stack((torch.ones_like(az), torch.zeros_like(az), torch.zeros_like(az)),0),
        torch.stack((torch.zeros_like(az), torch.cos(az), -torch.sin(az)),0),
        torch.stack((torch.zeros_like(az), torch.sin(az), torch.cos(az)),0)))
    ,0).to(device)
    # print('rormat',rotmat_az.shape)

    rotmat_el = torch.stack(
        ((torch.stack((torch.cos(el), torch.zeros_like(az), torch.sin(el)),0),
        torch.stack((torch.zeros_like(az), torch.ones_like(az), torch.zeros_like(az)),0),
        torch.stack((-torch.sin(el), torch.zeros_like(az), torch.cos(el)),0)))
    ,0).to(device)
    # print('rotmat_el',rotmat_el.shape)

    rotmat_az = rotmat_az.permute(2,0,1)
    # print('rotmat',rotmat_az)
    rotmat_el = rotmat_el.permute(2,0,1)
    rotmat = torch.matmul(rotmat_el, rotmat_az)#相同
    # print(rotmat.shape)

    t=torch.Tensor([tx,ty,-tz])

    tr_mat = torch.unsqueeze(t,0).repeat([batch_size,1]) # [B,3]
    tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
    tr_mat = tr_mat.permute(0,2,1) # [B,1,3]
    tr_mat = tr_mat.repeat([1,N_PTS,1]).to(device) # [B,1024,3]
    # print(xyz.permute(0,2,1).shape)
    xyz_out = torch.matmul(rotmat,xyz.permute(0,2,1)) - torch.matmul(rotmat, tr_mat.permute(0,2,1))
    # print('xyz_out', xyz.permute(0,2,1)- tr_mat.permute(0,2,1))
#对照下两个矩阵
    return xyz_out.permute(0,2,1).cuda()

def scale2one(p):
    x_min,_ = torch.min(p[..., 0],1)
    y_min,_ = torch.min(p[..., 1],1)
    z_min,_ = torch.min(p[..., 2],1)
    x_max,_ = torch.max(p[..., 0],1)
    y_max,_ = torch.max(p[..., 1],1)
    z_max,_ = torch.max(p[..., 2],1)

    x_center = (x_max+x_min)/2
    y_center = (y_max+y_min)/2
    z_center = (z_max+z_min)/2

    x_delta = x_max-x_min
    y_delta = y_max-y_min
    z_delta = z_max-z_min

    # x_center_cube = x_center/x_delta
    # y_center_cube = y_center/y_delta
    # z_center_cube = z_center/z_delta
    # print((p[..., 0]).shape)

    # p[..., 0] = p[..., 0] - torch.unsqueeze(x_center, 1)
    # p[..., 1] = p[..., 1] - torch.unsqueeze(y_center, 1)
    # p[..., 2] = p[..., 2] - torch.unsqueeze(z_center, 1)

    p[..., 0] = 2*(p[..., 0])/(abs(torch.unsqueeze(x_delta, 1)))
    p[..., 1] = 2*(p[..., 1])/(abs(torch.unsqueeze(y_delta, 1)))
    p[..., 2] = 2*(p[..., 2])/(abs(torch.unsqueeze(z_delta, 1)))

    # x_max,_ = torch.max(p[..., 0],1)
    # y_max,_ = torch.max(p[..., 1],1)
    # z_max,_ = torch.max(p[..., 2],1)
    # x_min,_ = torch.min(p[..., 0],1)
    # y_min,_ = torch.min(p[..., 1],1)
    # z_min,_ = torch.min(p[..., 2],1)
    return p

def average_pcl(p):
    x_mean = torch.mean(p[:, :, 0])
    y_mean = torch.mean(p[:, :, 1])
    z_mean = torch.mean(p[:, :, 2])
    p[:, :, 0] = p[:, :, 0] - x_mean
    p[:, :, 1] = p[:, :, 1] - y_mean
    p[:, :, 2] = p[:, :, 2] - z_mean
    return p, x_mean, y_mean, z_mean

def outlier(p, x_mean, y_mean, z_mean):

    threold = 0.01
    p_ = p
    # x_mean = torch.mean(p_[:, :, 0])
    x_max,index_x = torch.max(p_[..., 0],1)
    y_max,index_y = torch.max(p_[..., 1],1)
    z_max,index_z = torch.max(p_[..., 2],1)
    x_min,index_x_ = torch.min(p_[..., 0],1)
    y_min,index_y_ = torch.min(p_[..., 1],1)
    z_min,index_z_ = torch.min(p_[..., 2],1)

    for i in range(len(index_x)):
        (p_[..., 0][i])[index_x[i]] = x_mean
        x_max_, _ = torch.max(p_[..., 0], 1)
        if (x_max[i] - x_max_[i]) / x_mean > threold:
            p = p_

    p_ = p
    for i in range(len(index_y)):
        (p_[..., 1][i])[index_y[i]] = y_mean
        y_max_, _ = torch.max(p_[..., 1], 1)
        if (y_max[i] - y_max_[i]) / y_mean > threold:
            p = p_

    p_ = p
    for i in range(len(index_z)):
        (p_[..., 2][i])[index_z[i]] = z_mean
        z_max_, _ = torch.max(p_[..., 2], 1)
        if (z_max[i] - z_max_[i]) / z_mean > threold:
            p = p_

    p_ = p
    for i in range(len(index_x_)):
        (p_[..., 0][i])[index_x[i]] = x_mean
        x_min_, _ = torch.min(p_[..., 0], 1)
        if (x_max[i] - x_max_[i]) / x_mean > threold:
            p = p_

    p_ = p
    for i in range(len(index_y_)):
        (p_[..., 1][i])[index_y[i]] = y_mean
        y_min_, _ = torch.min(p_[..., 1], 1)
        if (y_max[i] - y_max_[i]) / y_mean > threold:
            p = p_
    p_ = p
    for i in range(len(index_z_)):
        (p_[..., 2][i])[index_z[i]] = z_mean
        z_min_, _ = torch.min(p_[..., 2], 1)
        if (z_min[i] - z_min_[i]) / z_mean > threold:
            p = p_
    return p