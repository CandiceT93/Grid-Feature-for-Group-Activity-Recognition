import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure
import torch

class motion_dataset_players(Dataset):  
    def __init__(self, filename_list, root_dir, mode, transform, frames_sequence):
 
        self.filename_list = filename_list
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.frames_sequence = frames_sequence

    def __len__(self):
        return len(self.filename_list)

    def stackopf(self, event_plyer):

        event_name = event_plyer.split(' ')[0]
        img_name = event_plyer.split(' ')[1]
        game_id = event_name.split('_')[0]
        event_path = os.path.join(self.root_dir, game_id, event_name)
        
        flow = torch.FloatTensor(20, 224, 224)

        for j, idx in enumerate(self.frames_sequence):
            frame_path = os.path.join(event_path, 'frame' + str('%06d' % idx), 'flow')
            u_video_path = os.path.join(frame_path, 'u')
            v_video_path = os.path.join(frame_path, 'v')

            h_image = os.path.join(u_video_path, img_name)
            v_image = os.path.join(v_video_path, img_name)
            
            imgH=(Image.open(h_image).convert('L'))
            imgV=(Image.open(v_image).convert('L'))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __getitem__(self, idx):

        event_plyer = self.filename_list[idx]
        data = self.stackopf(event_plyer)

        sample = (event_plyer, data)
           
        return sample

class motion_dataloader_players():
    def __init__(self, BATCH_SIZE, num_workers, path, img_list, frames_sequence):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.img_list = img_list
        self.frames_sequence = frames_sequence

    def run(self):
        self.filename_list = [line.strip() for line in open(self.img_list).readlines()]
        val_loader = self.validate()

        return val_loader, self.filename_list
                         

    def validate(self):
        validation_set = motion_dataset_players(filename_list = self.filename_list, root_dir=self.data_path, mode='test', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
                frames_sequence = self.frames_sequence)
        
        print '==> Validation data :',len(validation_set),'frames'
        #print validation_set[1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


