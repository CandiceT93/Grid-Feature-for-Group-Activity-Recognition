import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure

class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

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

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            # clips.append(random.randint(1, nb_clips/3))
            # clips.append(random.randint(nb_clips/3, nb_clips*2/3))
            # clips.append(random.randint(nb_clips*2/3, nb_clips+1))
            clips.append(26)
            
        elif self.mode == 'test':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        #label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='test':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, train_ucf_list, test_ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.train_frame_count ={}
        self.test_frame_count = {}
        self.train_video = {}
        self.test_video = {}
        self.train_ucf_list = train_ucf_list
        self.test_ucf_list = test_ucf_list

    def load_frame_count(self, split):

        if split == 'train':
        	split_list = self.train_ucf_list
        else:
        	split_list = self.test_ucf_list
        lines = [line.strip() for line in open(split_list).readlines()]

        for line in lines:

            videoname = line.split(' ')[0]
            label = int(line.split(' ')[1])
            num_imgs = int(line.split(' ')[2])
            video_path = self.data_path + videoname
            if split == 'train':
            	self.train_frame_count[videoname] = num_imgs
            	self.train_video[videoname] = label
            else:
            	self.test_frame_count[videoname] = num_imgs
            	self.test_video[videoname] = label


    def run(self):
        self.load_frame_count('train')
        self.load_frame_count('test')
        self.get_training_dic()
        self.val_sample()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}
        for video in self.train_video:
            #print videoname
            nb_frame = self.train_frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample(self):
        print '==> sampling testing frames'
        self.dic_testing={}
        for video in self.test_video:
            key = video + ' ' + str(26)
            self.dic_testing[key] = self.test_video[video]
            # nb_frame = self.test_frame_count[video]-10+1
            # interval = int(nb_frame/4)
            # for i in range(4):
            #     frame = i*interval
            #     key = video+ ' '+str(frame+1)
            #     self.dic_testing[key] = self.test_video[video]      

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                #transforms.RandomCrop(224),
		        transforms.Scale([224,224]),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(training_set),'frames'
        print training_set[1][0]['img0'].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='test', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print '==> Validation data :',len(validation_set),'frames'
        print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader


