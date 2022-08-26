from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(r"./metric/emd/")
sys.path.append(r"./metric/chamfer3D/")
sys.path.append(r"./models/")
sys.path.append(r"./utils/")
from repvgg_edge_nose_NEW_cmlp import generator
from torch.autograd import Variable
import cv2
import matplotlib.pylab as plt
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def save_img(category):

    NUM_VIEWS = 1
    cudnn.benchmark = True
    # azim = -50
    # elev = -145
    scale = 0.4
    base_path = '../data/data/shapenet/'
    save_path = './output/CGNet_VENet/Ours' + '/' +category + '/shapenet_img/'
    pickle_path = './output/repvgg_edge_nose_NEW_cmlp' + '/' + category + '/checkpoints/model_best.pth.tar'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)
    file_paths = os.listdir('./output/pic2points/' + 'ShapeNetRendering/' + category + '/')

    gen = generator()
    gen.cuda().eval()
    with open(pickle_path, "rb") as f:
        checkpoint = torch.load(f)
        gen.load_state_dict(checkpoint['state_dict'])

    for file_path in file_paths:
        point_path = base_path + 'ShapeNet_pointclouds/' + category + '/' + file_path + '/pointcloud_1024.npy'
        points_gt = (np.load(point_path)).astype('float32')  # groundtruth

        # save groundtruth img
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
        print('point_gt', points_gt.shape)
        for i in range(len(points_gt)):
            ax.scatter(points_gt[i, 1], points_gt[i, 2], points_gt[i, 0], c='r', s=5, depthshade=True)#00008B
        ax.axis('off')
        ax.view_init(azim=azim, elev=elev)
        plt.savefig(save_path + file_path + '_gt.png')
        plt.close()

        for i in range(NUM_VIEWS):
            img_path = './output/pic2points/ShapeNetRendering/03636649/e9f5e9f6f8d54caae455de02837313a6/rendering/' + (str(int(9)).zfill(2) + '.png')
            image = cv2.imread(img_path)[4:-5, 4:-5, :3]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)
            image = image.unsqueeze(0)

            image = Variable(image.float())
            image = image.cuda()
            _, _, points = gen(image)
            points = points.cpu().detach().numpy()
            points = np.squeeze(points)
            points = np.transpose(points, (1, 0))

            # save predict img
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_zlim(-scale, scale)
            for i in range(len(points)):
                ax.scatter(points[i, 1], points[i, 2], points[i, 0], c='r', s=5, depthshade=True)#'#00008B'1f77d4
            ax.axis('off')
            ax.view_init(azim=azim, elev=elev)
            plt.savefig(save_path + file_path + '_' + (str(int(9)).zfill(2) + '_pr2.png'))#i % NUM_VIEWS
            plt.close()

        print("saving model img...")

def main():

    cats = ['03636649']
    #    {"airplane":02691156, "bench": '02828884' "cabinet":02933112, "car":02958343, "lamp":03636649,
    #    "monitor":03211117, "rifle":04090263, "sofa":04256520, "speaker":03691459, "table":04379243,
    #    "telephone":04401088, "vessel":04530566, "chair":03001627}
    print(cats)
    for cat in cats:
        cat = str(cat)
        save_img(cat)

if __name__ == '__main__':
    main()
