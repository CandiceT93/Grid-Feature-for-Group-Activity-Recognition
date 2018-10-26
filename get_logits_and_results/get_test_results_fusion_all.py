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
from for_fusion_network import *
from for_fusion_ConvGrid import *

from sklearn.metrics import confusion_matrix

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=2, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--num_classes', default=6, type=int, metavar='N', help='number of classes in the dataset')
parser.add_argument('--checkpoint_path', default='./experiments_new/checkpoints/all_fusion_new/model_best.pth.tar', type=str, metavar='PATH', help='path for saving checkpoint')
parser.add_argument('--weight_per_class', default=[1, 1, 1, 0.1, 0.3, 1], type=list, metavar='WEIGHT', help='weight for each class when calculating loss')
parser.add_argument('--use_attention', default=True, type=bool, metavar='N', help='whether using attention')
parser.add_argument('--frames_list', default=[23, 24, 25, 26, 27, 28], type=list, metavar='FRAMES', help='the frames of optical flows used')
parser.add_argument('--output_path', default='./experiments_new/all_fusion_results_new', type=str, metavar='PATH', help='')
parser.add_argument('--output_score_name', default='score.txt', type=str, metavar='PATH', help='')
parser.add_argument('--output_video_name', default='video.txt', type=str, metavar='PATH', help='')
parser.add_argument('--output_label_name', default='label.txt', type=str, metavar='PATH', help='')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    if not os.path.exists(arg.output_path):
        os.makedirs(arg.output_path)
    w_video_level_preds = open(os.path.join(arg.output_path, arg.output_score_name), 'a')
    w_video_names = open(os.path.join(arg.output_path, arg.output_video_name), 'a')
    w_labels = open(os.path.join(arg.output_path, arg.output_label_name), 'a')

    #Prepare DataLoader
    data_loader = dataloader.grid_frame_rgb_flow_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        rgb_grid_path='/home/candice/Documents/dataset_icehockey/player_features/feature_arrays_new/',
                        rgb_frame_path='/home/candice/Documents/dataset_icehockey/events/flipped_data_correction/jpeg/',
                        flow_grid_path='/home/candice/Documents/dataset_icehockey/player_features/feature_arrays_motion_new/',
                        flow_frame_path='/home/candice/Documents/dataset_icehockey/events/flipped_data_correction/flow',
                        train_ucf_list ='/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/grid_feature_lists/new_train_list.txt',
                        test_ucf_list = '/home/candice/Documents/end-to-end-two-stream/two-stream-action-recognition-icehockey/grid_feature_lists/new_test_list.txt',
                        frames_list=arg.frames_list
                        )
    
    train_loader, test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        batch_size=arg.batch_size,
                        start_epoch=arg.start_epoch,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video, 
                        checkpoint_path = arg.checkpoint_path,
                        w_video_level_preds = w_video_level_preds,
                        w_video_names = w_video_names,
                        w_labels = w_labels
    )
    #Training
    cfm, acc = model.run()
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    print acc
    print cfm

class fusion_fc(nn.Module):
    def __init__(self, num_classes, use_attention):
        super(fusion_fc, self).__init__()
        self.model_frame_rgb = resnet101(pretrained = False, channel = 3, num_classes = num_classes).cuda()
        self.model_grid_rgb = ConvGrid(num_classes = num_classes, feature_dim = 2048, use_attention = use_attention).cuda()
        self.model_frame_flow = resnet101(pretrained = False, channel = 12, num_classes = num_classes).cuda()
        self.model_grid_flow = ConvGrid(num_classes = num_classes, feature_dim = 2048, use_attention = use_attention).cuda()
        self.fc = nn.Linear(256*2, num_classes)
    def forward(self, inputs_frame_rgb, inputs_grid_rgb, inputs_frame_flow, inputs_grid_flow):
        outputs_frame_rgb = self.model_frame_rgb(inputs_frame_rgb)
        outputs_grid_rgb = self.model_grid_rgb(inputs_grid_rgb)
        outputs_frame_flow = self.model_frame_flow(inputs_frame_flow)
        outputs_grid_flow = self.model_grid_flow(inputs_grid_flow)
        
        outputs = torch.cat((outputs_frame_rgb, outputs_grid_rgb, outputs_frame_flow, outputs_grid_flow), 1)
        outputs = self.fc(outputs)
        return outputs


class Spatial_CNN():
    def __init__(self, nb_epochs, batch_size, start_epoch, train_loader, test_loader, test_video,
        checkpoint_path, w_video_level_preds, w_video_names, w_labels):
        self.nb_epochs=nb_epochs
        self.batch_size=batch_size
        self.start_epoch=start_epoch
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.checkpoint_path = checkpoint_path
        self.w_video_level_preds = w_video_level_preds
        self.w_video_names = w_video_names
        self.w_labels = w_labels
        self.labels = []
        self.predictions = []

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = fusion_fc(num_classes=arg.num_classes, use_attention=arg.use_attention).cuda()
        checkpoint_arch = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint_arch['state_dict'])

        # self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        # #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    

    def run(self):
        self.build_model()
        #self.resume_and_evaluate()
        cudnn.benchmark = True
        prec1 = self.validate_1epoch()
        cfm = self.compute_and_save_cfm()
        return cfm

    def validate_1epoch(self):
        # print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys, rgb_grid_data, rgb_frame_data, flow_grid_data, flow_frame_data, label) in enumerate(progress):

            rgb_grid_data = rgb_grid_data.transpose(1, 3).transpose(2, 3)  
            flow_grid_data = flow_grid_data.transpose(1, 3).transpose(2, 3)
            rgb_grid_input_var = Variable(rgb_grid_data.float()).cuda()
            flow_grid_input_var = Variable(flow_grid_data.float()).cuda()
            rgb_frame_input_var = Variable(rgb_frame_data).cuda()
            flow_frame_input_var = Variable(flow_frame_data).cuda()
            output = self.model.forward(rgb_frame_input_var, rgb_grid_input_var, flow_frame_input_var, flow_grid_input_var)

            label_var = Variable(label, volatile=True).cuda(async=True)

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

        video_top1, video_top5, video_labels, video_preds, video_level_preds = self.frame2_video_level_accuracy()
        self.labels.append(video_labels)
        self.predictions.append(video_preds)
            

        # info = {'Epoch':[self.epoch],
        #         'Batch Time':[round(batch_time.avg,3)],
        #         'Loss':[round(video_loss,5)],
        #         'Prec@1':[round(video_top1,3)],
        #         'Prec@5':[round(video_top5,3)]}
        # record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),arg.num_classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
            preds = self.dic_video_level_preds[name]
#            label = int(self.test_video[name])-1
            video_name = name.split('_')[0] + '/' + name + '-frame000026.npy'
            label = (self.test_video[video_name])
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,2))
        # loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,video_level_labels, np.argmax(video_level_preds, axis=1), video_level_preds

    def compute_and_save_cfm(self):
        labels = self.labels[0].cpu().numpy()
        preds = self.predictions[0].cpu().numpy()
        acc = float(np.sum(labels == preds)) / float(labels.shape[0])
        cfm = confusion_matrix(labels, preds)
        return cfm, acc







if __name__=='__main__':
    main()
