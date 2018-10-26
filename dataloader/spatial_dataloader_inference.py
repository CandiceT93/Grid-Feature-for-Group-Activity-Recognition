import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure

class spatial_dataset_inference(Dataset):  
    def __init__(self, dic, root_dir, mode, transform, frames_sequence):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.frames_sequence = frames_sequence

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self, video_name, index):
         
        path = self.root_dir + video_name
        img = Image.open(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        try:
            transformed_img = self.transform(img)
        except:
            print(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        video_name= self.keys[idx]
        label = self.values[idx]
        #label = int(label)-1
        batch_data = []
        '''
        for index in self.frames_sequence:
            data = self.load_ucf_image(video_name,index)
            batch_data.append(data)
        '''
        batch_data = self.load_ucf_image(video_name, self.frames_sequence)
        sample = (video_name, batch_data, label)
           
        return sample

class spatial_dataloader_inference():
    def __init__(self, BATCH_SIZE, num_workers, path, test_ucf_list, ucf_split, frames_sequence):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.test_frame_count = {}
        self.test_video = {}
        self.test_ucf_list = test_ucf_list
        self.frames_sequence = frames_sequence

    def load_frame_count(self, split):

        split_list = self.test_ucf_list
        lines = [line.strip() for line in open(split_list).readlines()]

        for line in lines:

            videoname = line.split('-frame')[0].split('/')[1]
            label = int(line.split(' ')[1])
            num_imgs = 50

            video_path = self.data_path + videoname
            self.test_frame_count[videoname] = num_imgs
            self.test_video[videoname] = label


    def run(self):
        self.load_frame_count('test')
        val_loader = self.validate()

        return val_loader, self.test_video
                    
    # def val_sample(self):
    #     print '==> sampling testing frames'
    #     self.dic_testing={}
    #     for video in self.test_video:
    #         for frame in self.frames_sequence:
    #             key = video+ ' '+str(frame)
    #             self.dic_testing[key] = self.test_video[video]      

    def validate(self):
        validation_set = spatial_dataset_inference(dic=self.test_video, root_dir=self.data_path, mode='test', transform = transforms.Compose([
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


