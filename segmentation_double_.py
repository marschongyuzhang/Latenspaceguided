from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import socket
# PyTorch includes
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# Custom includes
from dataloaders.combine_dbs import CombineDBs as combine_dbs
import dataloaders.pascal as pascal
import dataloaders.sbd as sbd
from dataloaders import custom_transforms as tr
from networks.loss import class_cross_entropy_loss
from dataloaders.helpers import *
from networks.mainnetwork import *
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
import os
from convertmask.utils.methods import getMultiShapes
from PIL import Image
import time
global img_path
img_path = input("please input the path of image: ")



def process(image_name):
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU: {} '.format(gpu_id))

    resume_epoch = 100  # param test epoch
    nInputChannels = 5  # input channel

    modelName = 'IOG_pascal'  # use the pascal good for daily things. not for industry. pascal voc
    net = Network(nInputChannels=nInputChannels,
                  num_classes=1,
                  backbone='resnet101',  #
                  output_stride=16,
                  sync_bn=None,
                  freeze_bn=False)

    pretrain_dict = torch.load('IOG_PASCAL_SBD.pth')  # a model with class label and informations pretrain dict
    net.load_state_dict(pretrain_dict)
    net.eval()
    image = np.array(Image.open(image_name).convert('RGB'))
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = cv2.selectROI(im_rgb)
    image = image.astype(np.float32)
    # define the bb
    bbox = np.zeros_like(image[..., 0])
    bbox[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = 0.8
    # bbox[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] = 1
    void_pixels = 0.8 - bbox  # 1
    # void_pixels = 0.8 - bbox #1
    sample = {'image': image, 'gt': bbox, 'void_pixels': void_pixels}

    trns = transforms.Compose([
        tr.CropFromMask(crop_elems=('image', 'gt', 'void_pixels'), relax=30, zero_pad=True),
        tr.FixedResize(
            resolutions={'gt': None, 'crop_image': (512, 512), 'crop_gt': (512, 512), 'crop_void_pixels': (512, 512)},
            flagvals={'gt': cv2.INTER_LINEAR, 'crop_image': cv2.INTER_LINEAR, 'crop_gt': cv2.INTER_LINEAR,
                      'crop_void_pixels': cv2.INTER_LINEAR}),
        tr.IOGPoints(sigma=10, elem='crop_gt', pad_pixel=10),  # changeable
        tr.ToImage(norm_elem='IOG_points'),
        tr.ConcatInputs(elems=('crop_image', 'IOG_points')),
        tr.ToTensor()])

    tr_sample = trns(sample)
    inputs = tr_sample['concat'][None]
    outputs = net.forward(inputs)[-1]
    pred = np.transpose(outputs.data.numpy()[0, :, :, :], (1, 2, 0))
    pred = 1 / (1 + np.exp(-pred))  # changeable
    pred = np.squeeze(pred)
    gt = tens2image(tr_sample['gt'])
    bbox = get_bbox(gt, pad=30, zero_pad=True)
    result = crop2fullmask(pred, bbox, gt, zero_pad=True, relax=0, mask_relax=False)

    light = np.zeros_like(image)
    light[:, :, 2] = 255.  # or 1:2:
    #blending = image * result[..., None]  # Masken changable and big -> hell small -> dark
    blending = image * result[..., None]
    blending[blending > 255] = 255
    img = cv2.cvtColor(blending.astype(np.uint8), cv2.COLOR_RGB2BGR)


    cv2.imshow('resulting segmentation', img)

    #cv2.imwrite("./test_img/result.png",cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    mask_test=np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),dtype=np.uint8)
    #mask_test = np.array(Image.open("./test_img/result.png"), dtype=np.uint8)

    height=mask_test.shape[0]
    width=mask_test.shape[1]
    imagemask = np.zeros(mask_test.shape, dtype=float, order='C')
    for h in range(0, height):
        for w in range(0, width):
            if mask_test[h, w] != 0:
                imagemask[h, w] = 1
            else:
                imagemask[h, w] = 0
#
    imagemask = imagemask.astype(np.uint8)
    im = Image.fromarray(imagemask)
    classN = image_name.split("/")[-1] #whole name
    classN2 = classN.split("_")[0] #classname
    classN3 = classN.split('.')[0]
    im.save('./test_img/mask1/' + classN3 + '.png')


    imgPath = image_name
    maskPath = './test_img/mask1/' + classN3 + '.png'
    savePath = './test_img/json1'


    getMultiShapes.getMultiShapes(imgPath, maskPath, savePath)  # without yaml

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
def get_img_file(image_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(image_name):
        for image_name in filenames:
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                imagelist.append(os.path.join(parent, image_name))
        return imagelist

#def read_directory(directory_name):
#    for filename in os.listdir(r"./"+directory_name):
        #print(filename) 
#        img = cv2.imread(directory_name + "/" + filename)
#        array_of_img.append(img)
        #print(img)
#        print(array_of_img)
 
        
# about gui
if __name__ == '__main__':
#img_path = input("please input the path of image: ")
    img_paths = get_img_file(img_path)
    for i in img_paths:
        parser = argparse.ArgumentParser(description='Run class agnostic segmentation')
        img_path_i = i
#       read_directory(img_path)
        parser.add_argument('--image_name', type=str, default=img_path_i)   
        ###python .py --image /path  #help='path to target image' default='./test_img' previous used 
        args = parser.parse_args()
        process(args.image_name)
        print(args.image_name)
        time.sleep(1)
