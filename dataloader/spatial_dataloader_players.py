import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from skimage import io, color, exposure

class spatial_dataset_players(Dataset):  
    def __init__(self, filename_list, root_dir, mode, transform, frames_sequence):
 
        self.filename_list = filename_list
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.frames_sequence = frames_sequence

    def __len__(self):
        return len(self.filename_list)

    def load_ucf_image(self, img_name, index):
         
        #img = Image.open(os.path.join(path, 'frame' + str('%06d'%(index)) + '.jpg'))
        img = Image.open(img_name)
        try:
            transformed_img = self.transform(img)
        except:
            print(img_name)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        img_name = self.filename_list[idx]

        #frame_path = img_name.split(' ')[0] + 'frame' + str('%06d'%(frame_id))
        #imgname = img_name.split(' ')[0] + 'frame' + str('%06d'%(frame_id)) + img_name.split(' ')[1]
        data = self.load_ucf_image(img_name, idx)

        sample = (img_name, data)
           
        return sample

class spatial_dataloader_players():
    def __init__(self, BATCH_SIZE, num_workers, path, img_list, ucf_split, frames_sequence):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frames_sequence = frames_sequence
        self.img_list = img_list


    def run(self):
        self.filename_list = [line.strip() for line in open(self.img_list).readlines()]
        val_loader = self.validate()

        return val_loader, self.filename_list
                         

    def validate(self):
        validation_set = spatial_dataset_players(filename_list = self.filename_list, root_dir=self.data_path, mode='test', transform = transforms.Compose([
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


