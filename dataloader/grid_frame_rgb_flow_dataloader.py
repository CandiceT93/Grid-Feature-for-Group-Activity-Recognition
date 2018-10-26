import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
import numpy as np
import torch

class grid_frame_rgb_flow_dataset(Dataset):  
    def __init__(self, dic, rgb_grid_root_dir, rgb_frame_root_dir, flow_grid_root_dir, flow_frame_root_dir, mode, frames_list, transform):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.rgb_grid_root_dir = rgb_grid_root_dir
        self.rgb_frame_root_dir = rgb_frame_root_dir
        self.flow_grid_root_dir = flow_grid_root_dir
        self.flow_frame_root_dir = flow_frame_root_dir
        self.mode = mode
        self.transform = transform
        self.frames_list = frames_list
        self.in_channel = len(frames_list)
        self.img_rows = 224
        self.img_cols = 224

    def __len__(self):
        return len(self.keys)

    def load_flow(self, video_name):
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        #i = int(self.clips_idx)


        for j, idx in enumerate(self.frames_list):
            u_video_path = (self.flow_frame_root_dir + '/u/' + video_name)
            v_video_path = (self.flow_frame_root_dir + '/v/' + video_name)

            h_image = os.path.join(u_video_path, 'frame' + str('%06d'%(idx)) + '.jpg')
            v_image = os.path.join(v_video_path, 'frame' + str('%06d'%(idx)) + '.jpg')
            
            imgH=(Image.open(h_image).convert('L'))
            imgV=(Image.open(v_image).convert('L'))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def load_image(self, video_name, index):
         
        path = self.rgb_frame_root_dir + video_name
        img = Image.open(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        try:
            transformed_img = self.transform(img)
        except:
            print(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        img.close()

        return transformed_img


    def load_grid_array(self, root_dir, array_name):
         
        arr_name = os.path.join(root_dir, array_name)

        img = np.load(arr_name)

        return img

    def __getitem__(self, idx):

        rgb_array_name = self.keys[idx]
        flow_array_name = rgb_array_name.split('-frame')[0] + '.npy'
        img_name = rgb_array_name.split('/')[1].split('-frame')[0]
        label = self.values[idx]
        
        if self.mode=='train':
            rgb_grid_data = self.load_grid_array(self.rgb_grid_root_dir, rgb_array_name)
            rgb_frame_data = self.load_image(img_name, 26)
            flow_grid_data = self.load_grid_array(self.flow_grid_root_dir, flow_array_name)
            flow_frame_data = self.load_flow(img_name)
            sample = (rgb_grid_data, rgb_frame_data, flow_grid_data, flow_frame_data, label)
        elif self.mode=='test':
            rgb_grid_data = self.load_grid_array(self.rgb_grid_root_dir, rgb_array_name)
            rgb_frame_data = self.load_image(img_name, 26)
            flow_grid_data = self.load_grid_array(self.flow_grid_root_dir, flow_array_name)
            flow_frame_data = self.load_flow(img_name)
            sample = (img_name, rgb_grid_data, rgb_frame_data, flow_grid_data, flow_frame_data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class grid_frame_rgb_flow_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, rgb_grid_path, rgb_frame_path, flow_grid_path, flow_frame_path, train_ucf_list, test_ucf_list, frames_list):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.grid_data_path = rgb_grid_path
        self.frame_data_path = rgb_frame_path
        self.grid_flow_data_path = flow_grid_path
        self.frame_flow_data_path = flow_frame_path
        self.train_video = {}
        self.test_video = {}
        self.train_ucf_list = train_ucf_list
        self.test_ucf_list = test_ucf_list
        self.frames_list = frames_list


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
        training_set = grid_frame_rgb_flow_dataset(dic=self.train_video, 
                rgb_grid_root_dir=self.grid_data_path, rgb_frame_root_dir=self.frame_data_path,
                flow_grid_root_dir=self.grid_flow_data_path, flow_frame_root_dir=self.frame_flow_data_path,
                mode='train', frames_list = self.frames_list,
                transform = transforms.Compose([
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
        validation_set = grid_frame_rgb_flow_dataset(dic=self.test_video, 
                rgb_grid_root_dir=self.grid_data_path, rgb_frame_root_dir=self.frame_data_path, 
                flow_grid_root_dir=self.grid_flow_data_path, flow_frame_root_dir=self.frame_flow_data_path,
                mode='test', frames_list = self.frames_list, 
                transform = transforms.Compose([
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


