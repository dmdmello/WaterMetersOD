#python imports
import ast
import re
import shutil
import os
import math
import copy

#torch imports
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

#other imports
import pandas 
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_dict_from_string(string):
    #uses ast and re libraries to extract a dictionary described by a string
    try:
        dict_from_string = ast.literal_eval(re.search('({.+})', string).group(0))
        return dict_from_string
    except: 
        print("String format invalid")
        return string
    
def get_bb_from_location_dict(location_dict):
    #gets the bounding box that fits the polygon coordinates given by the 'data' key in a location dict
    data_dict_list = location_dict['data']
    x_coord = []
    y_coord = []
    for d in data_dict_list:
        x_coord.append(d['x'])
        y_coord.append(d['y'])
    x_min = np.min(x_coord)
    x_max = np.max(x_coord)
    y_min = np.min(y_coord)
    y_max = np.max(y_coord)
    return [x_min, y_min, x_max, y_max]


def draw_bb_from_coordinates(img, bb):
    #given an image ad bb coordinates, draw the bb over the image
    x_min, y_min, x_max, y_max = bb
    color_1 = [1]
    color_0 = [0,2]
    thic = 2
    #color intensity 1
    img[color_1,y_min-thic:y_max+thic, x_min-thic:x_min+thic] = 1.0
    img[color_1,y_min-thic:y_max+thic, x_max-thic:x_max+thic] = 1.0
    img[color_1,y_min-thic: y_min+thic, x_min-thic:x_max+thic] = 1.0
    img[color_1,y_max-thic: y_max+thic, x_min-thic:x_max+thic] = 1.0
    #color intensity 0
    img[color_0,y_min-thic:y_max+thic, x_min-thic:x_min+thic] = -1.0
    img[color_0,y_min-thic:y_max+thic, x_max-thic:x_max+thic] = -1.0
    img[color_0,y_min-thic: y_min+thic, x_min-thic:x_max+thic] = -1.0
    img[color_0,y_max-thic: y_max+thic, x_min-thic:x_max+thic] = -1.0
    
    
def show(img, rows):
    #prints pytorch image tensor
    npimg = img.detach().numpy()
    plt.figure(figsize = (20, rows))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.show()
    
def hflip_bb(bb_batch, img_size):
    #horizontal flip
    bb_batch[:,0] = 1-bb_batch[:,0] 
    bb_batch[:,2] = 1-bb_batch[:,2] 
    bb_batch = bb_batch[:,[2,1,0,3]]
    return bb_batch

def vflip_bb(bb_batch, img_size):
    #vertical flip
    bb_batch[:,1] = 1-bb_batch[:,1] 
    bb_batch[:,3] = 1-bb_batch[:,3] 
    bb_batch = bb_batch[:,[0,3,2,1]]
    return bb_batch

def rot90_bb(bb_batch, img_size):
    #90 degree rotation
    bb_batch[:,1] = 1 - bb_batch[:,1]
    bb_batch[:,3] = 1 - bb_batch[:,3]
    bb_batch = bb_batch[:,[3,0,1,2]]
    return bb_batch

def efficient_bb_aug(img_batch, bb_batch):
    #spatial augmentions on an image batch, performed in parallel, updating each bb accordingly
    batch_size = img_batch.shape[0]
    assert(bb_batch.shape[0] == batch_size)
    #efficient parallel horizontal flip random half of the batch
    random_idxs = torch.randperm(batch_size)
    slice_range = random_idxs[0:batch_size//2]
    img_batch[slice_range] = img_batch[slice_range].flip(-1)
    bb_batch[slice_range] = hflip_bb(bb_batch[slice_range], img_size=img_batch.shape[-1])
    #efficient parallel vertical flip for random half of the batch
    random_idxs = torch.randperm(batch_size)
    slice_range = random_idxs[0:batch_size//2]
    img_batch[slice_range] = img_batch[slice_range].flip(-2)
    bb_batch[slice_range] = vflip_bb(bb_batch[slice_range], img_size=img_batch.shape[-1])
    #efficient parallel 90 deg rotation for random half of the batch
    random_idxs = torch.randperm(batch_size)
    slice_range = random_idxs[0:batch_size//2]
    img_batch[slice_range] = torchvision.transforms.functional.rotate(img_batch[slice_range], -90) 
    bb_batch[slice_range] = rot90_bb(bb_batch[slice_range], img_size=img_batch.shape[-1])    
    
    return img_batch, bb_batch

def create_resized_imgs_folder(preprocessed_path, original_path, image_files, resized_dimensions = 256):
    #preprocesing steps for images in the TLKWaterMeters dataset
    resized_dimensions = (resized_dimensions, resized_dimensions)
    for i, img_name in enumerate(image_files):
        path = os.path.join(original_path, img_name)
        # Load the image in img variable
        img = cv2.imread(path, 1)
        # Create resized image using the calculated dimensions
        resized_image = cv2.resize(img, resized_dimensions, interpolation=cv2.INTER_AREA)
        # Save the image in Output Folder
        preprocessed_image_path = os.path.join(preprocessed_path, img_name)
        cv2.imwrite(preprocessed_image_path, resized_image)
    print("Images preprocessed Successfully")
    
def print_bb_prediction(model, dl, img_save_path, no_samples = 20, use_gpu=True):
    #samples some images from a dataloader and saves the images with the predicted bb
    x, _ = next(iter(dl))
    if use_gpu:
        x = x.cuda()
    no_samples = min(no_samples, x.shape[0])
    img_size = x.shape[-1]
    pred_bb = model(x)
    #pred_bb = bb_coord2_to_coord1(pred_bb).clip(0, 1)
    pred_bb = pred_bb.clip(0,1)
    pred_bb = (pred_bb*img_size).type(torch.int32)
    for bb, img in zip(pred_bb, x):
        draw_bb_from_coordinates(img, bb = bb)
    grid = make_grid(x[:no_samples].cpu(), nrow=10, normalize=True)
    #show(grid, 5)
    save_image(grid, img_save_path)
    
    
def get_iou_batch(target_bb, pred_bb):
    # find iou given pytorch tensor batches of targets bb coordinates and predicted coordinates
    with torch.no_grad():
        #coordinates of target bb
        xa_target = target_bb[:, 0]
        ya_target = target_bb[:, 1]
        xb_target = target_bb[:, 2]
        yb_target = target_bb[:, 3]
        #coordinates of predicted bb
        xa_pred = pred_bb[:, 0].clip(0, 1)
        ya_pred = pred_bb[:, 1].clip(0, 1)
        xb_pred = pred_bb[:, 2].clip(0, 1)
        yb_pred = pred_bb[:, 3].clip(0, 1)
        #coordinates of intersection bb
        xa_intsc = torch.max(xa_target, xa_pred)
        ya_intsc = torch.max(ya_target, ya_pred)
        xb_intsc = torch.min(xb_target, xb_pred)
        yb_intsc = torch.min(yb_target, yb_pred)
        #areas
        area_pred_bb =  (xb_pred - xa_pred).relu()*(yb_pred- ya_pred).relu()
        area_target_bb =  (xb_target - xa_target)*(yb_target- ya_target)
        area_intsc_bb = (xb_intsc - xa_intsc).relu()*(yb_intsc - ya_intsc).relu()
        area_union_bb = area_target_bb + area_pred_bb - (area_intsc_bb)
        #final iou computation
        iou_batch = ((area_intsc_bb/area_union_bb))
    return iou_batch