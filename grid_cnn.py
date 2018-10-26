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
from ConvGrid import *
from ConvGrid_Simple import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--num_classes', default=6, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--pretrained', default=False, type=bool, metavar='N', help='whether to load pretrained model')
parser.add_argument('--checkpoint_path', default='./experiments_new/checkpoints/spatial_grid_simple', type=str, metavar='PATH', help='path for saving checkpoint')
parser.add_argument('--weight_per_class', default=[1, 1, 1, 0.1, 0.3, 1], type=list, metavar='WEIGHT', help='weight for each class when calculating loss')
parser.add_argument('--spatial', default=True, type=bool, metavar='N', help='whether spatial feature')
parser.add_argument('--use_attention', default=True, type=bool, metavar='N', help='whether using attention')


def main():
    global arg
    arg = parser.parse_args()
    print arg
    if arg.checkpoint_path:
        if not os.path.exists(arg.checkpoint_path):
            os.makedirs(arg.checkpoint_path)
    if arg.weight_per_class:
        weights = np.array(arg.weight_per_class)
        weights = torch.FloatTensor(weights)
    #Prepare DataLoader
    if arg.spatial == True:
        data_loader = dataloader.grid_dataloader(
                            BATCH_SIZE=arg.batch_size,
                            num_workers=8,
                            path='/home/candice/Documents/dataset_icehockey/player_features/feature_arrays_new/',
                            train_ucf_list ='/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/grid_feature_lists/new_train_list.txt',
                            test_ucf_list = '/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/grid_feature_lists/new_test_list.txt',
                            )
    else:
        data_loader = dataloader.grid_dataloader(
                            BATCH_SIZE=arg.batch_size,
                            num_workers=8,
                            path='/home/candice/Documents/dataset_icehockey/player_features/feature_arrays_motion_new/',
                            train_ucf_list ='/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/grid_feature_lists_motion/new_train_list.txt',
                            test_ucf_list = '/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/grid_feature_lists_motion/new_test_list.txt',
                            )

    
    train_loader, test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video, 
                        weights = weights
    )
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video, weights):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.weights = weights

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = ConvGrid_Simple(num_classes=arg.num_classes, feature_dim=2048, use_attention=arg.use_attention).cuda()
        #Loss function and optimizer
        if arg.weight_per_class:
            self.criterion = nn.CrossEntropyLoss(weight=self.weights).cuda()
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,os.path.join(arg.checkpoint_path, 'checkpoint.pth.tar'), os.path.join(arg.checkpoint_path, 'model_best.pth.tar'))

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            # data_dict shape [batch_size, 8, 12, 2048]
            data = data_dict.transpose(1, 3).transpose(2, 3)

            #print data.shape
            #raise Exception("here")

            # output = Variable(torch.zeros(len(data_dict['img0']),arg.num_classes).float()).cuda()
            # for i in range(len(data_dict)):
            #     key = 'img'+str(i)
            #     data = data_dict[key]
            #     input_var = Variable(data).cuda()
            #     output += self.model(input_var)

            input_var = Variable(data.float()).cuda()
            output = self.model(input_var)
            #print output.shape
            #raise Exception("here")

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            #prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            prec1, prec2 = accuracy(output.data, label, topk=(1, 2))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            #top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
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
            data = data.transpose(1, 3).transpose(2, 3)
            data_var = Variable(data.float(), volatile=True).cuda(async=True)
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
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
            

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
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
        return top1,top5,loss.data.cpu().numpy()







if __name__=='__main__':
    main()
