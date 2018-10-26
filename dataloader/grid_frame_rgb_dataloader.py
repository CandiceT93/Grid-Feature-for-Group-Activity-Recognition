import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
import numpy as np

class grid_frame_rgb_dataset(Dataset):  
    def __init__(self, dic, grid_root_dir, frame_root_dir, mode, transform):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.grid_root_dir = grid_root_dir
        self.frame_root_dir = frame_root_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_image(self, video_name, index):
         
        path = self.frame_root_dir + video_name
        img = Image.open(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        try:
            transformed_img = self.transform(img)
        except:
            print(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        img.close()

        return transformed_img


    def load_grid_array(self, array_name):
         
        arr_name = os.path.join(self.grid_root_dir, array_name)

        img = np.load(arr_name)

        return img

    def __getitem__(self, idx):

        array_name = self.keys[idx]
        video_name = array_name.split('/')[1].split('-frame')[0]
        label = self.values[idx]
        
        if self.mode=='train':
            grid_data = self.load_grid_array(array_name)
            frame_data = self.load_image(video_name, 26)
            sample = (grid_data, frame_data, label)
        elif self.mode=='test':
            grid_data = self.load_grid_array(array_name)
            frame_data = self.load_image(video_name, 26)
            sample = (video_name, grid_data, frame_data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class grid_frame_rgb_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, grid_path, frame_path, train_ucf_list, test_ucf_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.grid_data_path=grid_path
        self.frame_data_path = frame_path
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
        training_set = grid_frame_rgb_dataset(dic=self.train_video, grid_root_dir=self.grid_data_path, frame_root_dir=self.frame_data_path, mode='train', transform = transforms.Compose([
                #transforms.RandomCrop(224),
                transforms.Scale([224,224]),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
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
        validation_set = grid_frame_rgb_dataset(dic=self.test_video, grid_root_dir=self.grid_data_path, frame_root_dir=self.frame_data_path, mode='test', transform = transforms.Compose([
                #transforms.RandomCrop(224),
                transforms.Scale([224,224]),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        # print '==> Validation data :',len(validation_set),'frames'
        # print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


