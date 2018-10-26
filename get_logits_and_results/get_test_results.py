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
from network import *

from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--checkpoint_path', default='./experiments_new/checkpoints/motion_frame/pretrained_14/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes', default=6, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--num_channels', default=12, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--frames_list', default=[23, 24, 25, 26, 27, 28], type=list, metavar='FRAMES', help='the frames of optical flows used')
parser.add_argument('--output_score_name', default='score.txt', type=str, metavar='PATH', help='')
parser.add_argument('--output_video_name', default='video.txt', type=str, metavar='PATH', help='')
parser.add_argument('--output_label_name', default='label.txt', type=str, metavar='PATH', help='')
parser.add_argument('--output_path', default='./outputs', type=str, metavar='PATH', help='')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    #path_prefix = '/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/for_fusion_new'
    if not os.path.exists(arg.output_path):
        os.makedirs(arg.output_path)
    w_video_level_preds = open(os.path.join(arg.output_path, arg.output_score_name), 'a')
    w_video_names = open(os.path.join(arg.output_path, arg.output_video_name), 'a')
    w_labels = open(os.path.join(arg.output_path, arg.output_label_name), 'a')

    #Prepare DataLoader
    if arg.num_channels == 3:
    	data_loader = dataloader.spatial_dataloader(
	                        BATCH_SIZE=arg.batch_size,
	                        num_workers=8,
	                        path='/home/candice/Documents/dataset_icehockey/events/flipped_data_correction/jpeg/',
	                        train_ucf_list ='/home/candice/Documents/dataset_icehockey/events/experiments/new/train_list.txt',
	                        test_ucf_list = '/home/candice/Documents/dataset_icehockey/events/experiments/new/test_list.txt',
	                        ucf_split ='01', 
	                        )
    else:
    	data_loader = dataloader.Motion_DataLoader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='/home/candice/Documents/dataset_icehockey/events/flipped_data_correction/flow',
                        train_ucf_list ='/home/candice/Documents/dataset_icehockey/events/experiments/new/train_list.txt',
                        test_ucf_list = '/home/candice/Documents/dataset_icehockey/events/experiments/new/test_list.txt',
                        ucf_split='01',
                        in_channel=arg.num_channels//2,
                        frames_list = arg.frames_list
                        )
    
    train_loader, test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        batch_size=arg.batch_size,
                        checkpoint_path=arg.checkpoint_path,
                        test_loader=test_loader,
                        test_video=test_video,
                        w_video_level_preds = w_video_level_preds,
                        w_video_names = w_video_names,
                        w_labels = w_labels
    )
    #model.build_model()
    #Training
    cfm = model.run()
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    print cfm

class Spatial_CNN():
    def __init__(self, batch_size, checkpoint_path,test_loader, test_video, w_video_level_preds, w_video_names, w_labels):
        self.batch_size=batch_size
        self.checkpoint_path=checkpoint_path
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.video_names = []
        self.labels = []
        self.predictions = []
        self.num_test = len(test_video)
        self.lr = 1e-3
        self.w_video_level_preds = w_video_level_preds
        self.w_video_names = w_video_names
        self.w_labels = w_labels

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
        prec1, val_loss = self.validate_1epoch()
        return

    def run(self):
		cudnn.benchmark = True
		self.build_model()
		self.resume_and_evaluate()
		cfm = self.compute_confusion_matrix()
		return cfm
        
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
        for i, (keys,data,label) in enumerate(progress):
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j]#.split('/',1)[0]
                self.w_video_names.write('%s\n' % videoName)
                self.w_labels.write('%s\n' % label[j])
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]
                np.savetxt(self.w_video_level_preds, preds[j, :])

        video_top1, video_top5, video_loss, video_labels, video_preds, video_level_preds = self.frame2_video_level_accuracy()
        self.labels.append(video_labels)
        self.predictions.append(video_preds)
            

        info = {'Epoch':[1],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        #record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),arg.num_classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
            preds = self.dic_video_level_preds[name]
#            label = int(self.test_video[name])-1
            label = (self.test_video[name])
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,2))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy(), video_level_labels, np.argmax(video_level_preds, axis = 1), video_level_preds



    def compute_confusion_matrix(self):
    	labels = self.labels[0].cpu().numpy()
    	preds = self.predictions[0].cpu().numpy()
    	print labels.shape, preds.shape
    	# self.labels = np.array(self.labels[0].cpu().numpy()).reshape(self.num_test, 1)
    	# self.preds = np.array(self.predictions[0].cpu().numpy()).reshape(self.num_test, 1)
    	cfm = confusion_matrix(labels, preds)
    	return cfm





if __name__=='__main__':
    main()