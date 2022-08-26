# coding: utf-8
from __future__ import print_function
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(r"../loss")
sys.path.append(r"../metric/emd/")
sys.path.append(r"../metric/chamfer3D/")
sys.path.append(r"../models/")
sys.path.append(r"../utils/")
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms

from repvgg_edge_nose_NEW_cmlp import generator

# 训练过的模型路径
resume_path = '../output/repvgg_edge_nose_NEW_cmlp/02691156/checkpoints/model_best.pth.tar'
# 输入图像路径
single_img_path = '../data/shapenet/ShapeNetRendering/02691156/10155655850468db78d106ce0a280f87/rendering/00.png'
# 绘制的热力图存储路径
save_path = '../output/repvgg_edge_nose_NEW_cmlp/02691156/temp_layer4.jpg'

# 网络层的层名列表, 需要根据实际使用网络进行修改
layers_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc', 'fc1', 'fc2', 'fc3', 'th']
# 指定层名
out_layer_name = "layer4"

features_grad = 0


# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False, out_layer=None):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    # 读取图像并预处理
    global layer2
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img).cuda()
    img = img.unsqueeze(0)  # (1, 3, 448, 448)

    # model转为eval模式
    model.eval()

    # 获取模型层的字典
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}
    for i, (name, module) in enumerate(model.features._modules.items()):
        layers_dict[layers_names[i]] = module

    # 遍历模型的每一层, 获得指定层的输出特征图
    # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
    features = img
    start_flatten = False
    features_flatten = None
    for name, layer in layers_dict.items():
        if name != out_layer and start_flatten is False:  # 指定层之前
            features = layer(features)
        elif name == out_layer and start_flatten is False:  # 指定层
            features = layer(features)
            start_flatten = True
        else:  # 指定层之后
            if features_flatten is None:
                features_flatten = layer(features)
            else:
                features_flatten = layer(features_flatten)

    features_flatten = torch.flatten(features_flatten, 1)
    output = model.classifier(features_flatten)

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output, 1).item()
    pred_class = output[:, pred]

    # 求中间变量features的梯度
    # 方法1
    # features.register_hook(extract)
    # pred_class.backward()
    # 方法2
    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]

    grads = features_grad  # 获取梯度
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    print("pooled_grads:", pooled_grads.shape)
    print("features:", features.shape)
    # features.shape[0]是指定层feature的通道数
    for i in range(features.shape[0]):
        features[i, ...] *= pooled_grads[i, ...]

    # 计算heatmap
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # 构建模型并加载预训练参数
    gen = generator().cuda()
    gen = nn.DataParallel(gen).cuda()
    checkpoint = torch.load("../output/repvgg_edge_nose_NEW_cmlp/02691156/checkpoints/model_best.pth.tar")
    gen.load_state_dict(checkpoint['state_dict'], False)

    draw_CAM(gen, single_img_path, save_path, transform=transform, visual_heatmap=True, out_layer=out_layer_name)