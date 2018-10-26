import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network_new import *

import os

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--checkpoint_path', default='./checkpoint_motion_flipped/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes', default=4, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--num_channels', default=12, type=int, metavar='N', help='number of channels')
parser.add_argument('--frames_list', default=[23, 24, 25, 26, 27, 28], type=list, metavar='FRAMES', help='the frames of optical flows used')
parser.add_argument('--frames_list_offset', default=9, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--output_path', default='./lstm_motion_test_flipped', type=str, metavar='PATH', help='path for saving output logits')
parser.add_argument('--frames_sequence', default=[23, 24, 25, 26, 27, 28], type=list, metavar='FRAMES', help='the frames of optical flows used')


def main():
    global arg
    arg = parser.parse_args()
    print arg

    if not os.path.exists(arg.output_path):
        os.makedirs(arg.output_path)
    wlogits = open(os.path.join(arg.output_path, 'logits.txt'), "a")
    wlabels = open(os.path.join(arg.output_path, 'labels.txt'), "a")
    wpreds = open(os.path.join(arg.output_path, 'preds.txt'), "a")

    #Prepare DataLoader
    if arg.num_channels == 3:
    	data_loader = dataloader.spatial_dataloader_inference(
	                        BATCH_SIZE=arg.batch_size,
	                        num_workers=8,
	                        path='/home/candice/Documents/dataset_icehockey/events/original_size/flow/',
	                        test_ucf_list = '/home/candice/Documents/dataset_icehockey/events/experiments/dumpin_dumpout_shot_pass/test_list.txt',
	                        ucf_split ='01', 
                            frames_sequence = arg.frames_sequence
	                        )
    else:
    	data_loader = dataloader.Motion_DataLoader_inference(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='/home/candice/Documents/dataset_icehockey/events/flipped_data_correction/flow',
                        test_ucf_list = '/home/candice/Documents/dataset_icehockey/events/experiments/dumpin_dumpout_shot_pass/test_list.txt',
                        frames_list = arg.frames_list,
                        offset = arg.frames_list_offset, 
                        )
    
    test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        batch_size=arg.batch_size,
                        checkpoint_path=arg.checkpoint_path,
                        test_loader=test_loader,
                        test_video=test_video,
                        wlogits = wlogits,
                        wlabels = wlabels,
                        wpreds = wpreds,
                        frames_sequence = arg.frames_sequence,
                        frames_list = arg.frames_list
    )
    #model.build_model()
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, batch_size, checkpoint_path,test_loader, test_video, wlogits, wlabels, wpreds, frames_sequence, frames_list):
        self.batch_size=batch_size
        self.checkpoint_path=checkpoint_path
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.num_test = len(test_video)
        self.lr = 1e-3
        self.wlogits = wlogits
        self.wlabels = wlabels
        self.wpreds = wpreds
        self.frames_sequence = frames_sequence
        self.frames_list = frames_list

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet50(pretrained = False, channel=arg.num_channels, num_classes=arg.num_classes).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if os.path.isfile(self.checkpoint_path):
            print("==> loading checkpoint '{}'".format(self.checkpoint_path))
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
              .format(self.checkpoint_path, checkpoint['epoch'], self.best_prec1))
        else:
            print("==> no checkpoint found at '{}'".format(self.checkpoint_path))
        self.validate_1epoch()
        return

    def run(self):
        cudnn.benchmark = True
        self.build_model()
        self.resume_and_evaluate()
        
    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(1, 1))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        num_acc = 0
        for i, (keys,data,label) in enumerate(progress):
            #print batch_data
            #raise Exception("here")
            np.savetxt(self.wlabels, label)
            label = label.cuda(async=True)

            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output, logits = self.model(data_var)
            logits = logits.data.cpu().numpy()
            np.savetxt(self.wlogits, logits)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            predictions = output.data.cpu().numpy()
            predictions = np.argmax(predictions, axis = 1)
            #if self.frames_list[0] + j == self.frames_sequence[0]:
            np.savetxt(self.wpreds, predictions)
            




if __name__=='__main__':
    main()