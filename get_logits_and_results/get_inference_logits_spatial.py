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
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--checkpoint_path', default='./experiments_new/checkpoints/spatial_frame/pretrained_54/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes', default=6, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--num_channels', default=3, type=int, metavar='N', help='number of channels')
parser.add_argument('--frames_list', default=[20, 21, 22, 23, 24, 25], type=list, metavar='FRAMES', help='the frames of optical flows used')
parser.add_argument('--frames_list_offset', default=9, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--frames_sequence', default=23, type=int, metavar='FRAMES', help='the frames of optical flows used')
parser.add_argument('--output_path', default='./experiments_new/for_lstm/test_logits', type=str, metavar='PATH', help='path for saving output logits')


def main():
    global arg
    arg = parser.parse_args()
    print arg

    if not os.path.exists(arg.output_path):
        os.makedirs(arg.output_path)
    wlogits = open(os.path.join(arg.output_path, 'logits.txt'), "a")
    wnames = open(os.path.join(arg.output_path, 'names.txt'), "a")

    #Prepare DataLoader
    if arg.num_channels == 3:
    	data_loader = dataloader.spatial_dataloader_inference(
	                        BATCH_SIZE=arg.batch_size,
	                        num_workers=8,
	                        path='/home/candice/Documents/dataset_icehockey/events/flipped_data_correction/jpeg/',
	                        test_ucf_list = './grid_feature_lists/new_train_list.txt',
	                        ucf_split ='01', 
                            frames_sequence = arg.frames_sequence
	                        )
    else:
    	data_loader = dataloader.Motion_DataLoader_inference(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='/home/candice/Documents/dataset_icehockey/events/original_size/flow',
                        test_ucf_list = '/home/candice/Documents/dataset_icehockey/events/experiments/new/test_list.txt',
                        ucf_split='01',
                        in_channel=arg.num_channels//2,
                        frames_list = arg.frames_list,
                        offset = arg.frames_list_offset
                        )
    
    test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        batch_size=arg.batch_size,
                        checkpoint_path=arg.checkpoint_path,
                        test_loader=test_loader,
                        test_video=test_video,
                        wlogits = wlogits,
                        wnames = wnames,
                        frames_sequence = arg.frames_sequence
    )
    #model.build_model()
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, batch_size, checkpoint_path,test_loader, test_video, wlogits, wnames, frames_sequence):
        self.batch_size=batch_size
        self.checkpoint_path=checkpoint_path
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.num_test = len(test_video)
        self.lr = 1e-3
        self.wlogits = wlogits
        self.wnames = wnames
        self.frames_sequence = frames_sequence

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained = False, channel=arg.num_channels, num_classes=arg.num_classes).cuda()
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
        for i, (keys,batch_data,label) in enumerate(progress):
            for j in range(len(keys)):
                self.wnames.write('%s\n' % keys[j])
            label = label.cuda(async=True)

            data_var = Variable(batch_data, volatile=True).cuda(async=True)
            # compute output
            output, logits = self.model(data_var)
            logits = logits.data.cpu().numpy()
            np.savetxt(self.wlogits, logits)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            




if __name__=='__main__':
    main()