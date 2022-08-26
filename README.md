# 3D-FENet

## Dataset

### ShapeNet
We train and validate our model on the ShapeNet dataset. We use the rendered images from the dataset provided by <a href="https://github.com/chrischoy/3D-R2N2" target="_blank" >3d-r2n2</a>, which consists of 13 object categories. For generating the ground truth point clouds, we sample points on the corresponding object meshes from ShapeNet. We use the dataset split provided by r2n2 in all the experiments. Data download links are provided below:<br>
Rendered Images (~12.3 GB): http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz <br>
ShapeNet pointclouds (~2.8 GB): https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g <br>
ShapeNet train/val split: https://drive.google.com/open?id=10FR-2Lbn55POB1y47MJ12euvobi6mgtc

Download each of the folders, extract them and move them into *data/shapenet/*.<br>
The folder structure should now look like this:<br>
--data/shapenet/<br>
&nbsp;&nbsp;--ShapeNetRendering/<br>
&nbsp;&nbsp;--ShapeNet_pointclouds/<br>
&nbsp;&nbsp;--splits/<br>

### Pix3D
We evaluate the generalization capability of our model by testing it on the real-world <a href="https://github.com/xingyuansun/pix3d">pix3d dataset</a>. For the ground truth point clouds, we sample 1024 points on the provided meshes. Data download links are provided below:<br>
Pix3D dataset (~20 GB): Follow the instructions in https://github.com/xingyuansun/pix3d <br>
Pix3D pointclouds (~13 MB): https://drive.google.com/open?id=1RZakyBu9lPbG85SyconBn4sR8r2faInV

Download each of the folders, extract them and move them into *data/pix3d/*.<br>
The folder structure should now look like this:<br>
--data/pix3d/<br>
&nbsp;&nbsp;--img_cleaned_input/<br>
&nbsp;&nbsp;--img/<br>
&nbsp;&nbsp;--mask/<br>
&nbsp;&nbsp;--model/<br>
&nbsp;&nbsp;--pix3d_pointclouds/<br>
&nbsp;&nbsp;--pix3d.json<br>

## Usage
Install [Pyrotch](https://pytorch.org/get-started/previous-versions/). We recommend pytorch. The code provided has been tested with Python 3.6.13, torch 1.5, torchvision 0.6.0+cu101, tensorboard 1.14.0, and CUDA 10.1. The following steps need to be performed to run the codes given in this repository:

1. Clone the repository:
```shell
git clone https://github.com/sunhui-3D/3D-FHNet.git
cd 3d-FENet
```
2. Pytorch for losses (Chamfer and EMD)  need to be setup. Run the setup as given below. (Note that the the nvcc, cudalib):
[Pytorch Chamfer Distance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git)

3.Install the requirements
```
pip install -r ./requirements.txt

## Training
- To train the network, run:
```shell
python train.py
```
##Testing
-To test the network,run
```shell
python test.py
```

## Trained Models


## Evaluation
Follow the steps detailed above to download the dataset and pre-trained models.


