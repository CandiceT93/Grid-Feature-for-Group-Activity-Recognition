import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from split_train_test_video import *
import os
 
class motion_dataset_inference_multiframes(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform, frames_stack):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224
        self.frames_stack = frames_stack

    def stackopf(self, frames_list):
        
        flow = torch.FloatTensor(self.in_channel,self.img_rows,self.img_cols)

        for j, idx in enumerate(frames_list):
            u_video_path = (self.root_dir + '/u/' + self.video)
            v_video_path = (self.root_dir + '/v/' + self.video)

            h_image = os.path.join(u_video_path, 'frame' + str('%06d'%(idx)) + '.jpg')
            v_image = os.path.join(v_video_path, 'frame' + str('%06d'%(idx)) + '.jpg')
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)

        self.video = self.keys[idx]

        label = self.values[idx]
        #label = int(label)-1
        batch_data = []
        for frames_list in self.frames_stack: 
            data = self.stackopf(frames_list)
            batch_data.append(data)

        if self.mode == 'train':
            sample = (batch_data,label)
        elif self.mode == 'val':
            sample = (self.video,batch_data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample





class Motion_DataLoader_inference_multiframes():
    def __init__(self, BATCH_SIZE, num_workers, path, test_ucf_list, frames_list, offset):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.in_channel = 2*len(frames_list)
        self.data_path=path
        self.test_video = {}
        self.test_ucf_list = test_ucf_list
        self.frames_list = frames_list
        self.offset = offset

        self.frames_stack = []
        for i in range(self.offset):
            tmp = [frame+i for frame in frames_list]
            self.frames_stack.append(tmp)
        

    def run(self):
        self.get_dic()
        val_loader = self.val()

        return val_loader, self.test_video
            
    def get_dic(self):
        lines = [line.strip() for line in open(self.test_ucf_list).readlines()]

        for line in lines:
            videoName = line.split('-frame')[0].split('/')[1]
            label = int(line.split(' ')[1])
            self.test_video[videoName] = label
             

    def val(self):
        validation_set = motion_dataset_inference_multiframes(dic = self.test_video, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]),
            frames_stack = self.frames_stack)
        #print '==> Validation data :',len(validation_set),' frames',validation_set[1][1].size()
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader
