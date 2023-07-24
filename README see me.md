
> [Interactive Object Segmentation with Inside-Outside Guidance](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Interactive_Object_Segmentation_With_Inside-Outside_Guidance_CVPR_2020_paper.pdf)  
> Shiyin Zhang, Jun Hao Liew, Yunchao Wei, Shikui Wei, Yao Zhao  

![img](https://github.com/shiyinzhang/Inside-Outside-Guidance/blob/master/ims/ims.png "img")

1. Usage
You can start training with the following commands:
```
# training step
python train.py
python train_refinement.py

# testing step
python test.py
python test_refinement.py

# train step
python eval.py
python eval_refinement.py
```
We set the paths of PASCAL/SBD dataset and pretrained model in mypath.py.

### Pretrained models
| Network |Dataset | Backbone |      Download Link        |
|---------|---------|-------------|:-------------------------:|
|IOG |PASCAL + SBD  |  ResNet-101 |  [IOG_PASCAL_SBD.pth](https://drive.google.com/file/d/1Lm1hhMhhjjnNwO4Pf7SC6tXLayH2iH0l/view?usp=sharing)     |
|IOG |PASCAL |  ResNet-101   |  [IOG_PASCAL.pth](https://drive.google.com/file/d/1GLZIQlQ-3KUWaGTQ1g_InVcqesGfGcpW/view?usp=sharing)   |
|IOG-Refinement |PASCAL + SBD  |  ResNet-101 |  [IOG_PASCAL_SBD_REFINEMENT.pth](https://drive.google.com/file/d/1VdOFUZZbtbYt9aIMugKhMKDA6EuqKG30/view?usp=sharing)     |

### Dataset
With the annotated bounding boxes (∼0.615M) of ILSVRCLOC, we apply our IOG to collect their pixel-level annotations, named Pixel-ImageNet, which are publicly available at https://github.com/shiyinzhang/Pixel-ImageNet.

1. pytorch opencv pycocotools scipy(1.2.1)
2. dataset pascal vox2012
3. Change the Path: mypath.py change the database return in your path in compute.
4. python segmentation_double.py --image_name E:/Datasets/OTB2015/Lemming/img/name of image.dataform

#### segmentatnion double: 
pycocotools torchvision = 0.2 python >= 3.5 PyTorch = 0.4 spicy (1.1.10) 
python segmentation_double.py
input: ./test_img/n02992211
then click the boundary and the box
then get the mask
then enter
then wait for 1s
then the next image.

