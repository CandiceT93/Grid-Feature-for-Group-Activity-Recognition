import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
import numpy as np

class grid_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name):
         
        arr_name = os.path.join(self.root_dir, video_name)

        img = np.load(arr_name)

        return img

    def __getitem__(self, idx):

        video_name = self.keys[idx]
        label = self.values[idx]
        
        if self.mode=='train':
            data = self.load_ucf_image(video_name)
            sample = (data, label)
        elif self.mode=='test':
            data = self.load_ucf_image(video_name)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class grid_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, train_ucf_list, test_ucf_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.train_video = {}
        self.test_video = {}
        self.train_ucf_list = train_ucf_list
        self.test_ucf_list = test_ucf_list


    def run(self):
        self.get_train_dic()
        self.get_val_dic()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_train_dic(self):
        lines = [line.strip() for line in open(self.train_ucf_list).readlines()]
        for line in lines:
            arr_name = line.split(' ')[0]
            arr_label = int(line.split(' ')[1])
            self.train_video[arr_name] = arr_label
                    
    def get_val_dic(self):
        lines = [line.strip() for line in open(self.test_ucf_list).readlines()]
        for line in lines:
            arr_name = line.split(' ')[0]
            arr_label = int(line.split(' ')[1])
            self.test_video[arr_name] = arr_label
 

    def train(self):
        training_set = grid_dataset(dic=self.train_video, root_dir=self.data_path, mode='train')
        # print '==> Training data :',len(training_set),'frames'
        # print type(training_set)
        # print training_set[1][0]['img0'].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = grid_dataset(dic=self.test_video, root_dir=self.data_path, mode='test')
        
        # print '==> Validation data :',len(validation_set),'frames'
        # print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


