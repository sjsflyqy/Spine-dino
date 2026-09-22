import os
import torch
from PIL import Image
import numpy as np
from skimage.measure import regionprops
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
from utils import *

def get_data_loader(dataset,data_path,mode='Training',b=4):
    if dataset == 'NHANESII_Vertebrae':
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        dataset = NHANESII(data_path=data_path,transforms=transform,mode=mode)
        data_loader = DataLoader(dataset, batch_size=b, shuffle=True, num_workers=4, collate_fn=collate_fn)
        return data_loader
    
    elif dataset == 'ResNet_training':
        dataset = ResNet_training(data_path=data_path,transforms=None,mode=mode)
        data_loader = DataLoader(dataset,batch_size=b,shuffle=True,num_workers=4, collate_fn=collate_fn)
        return data_loader

class ResNet_training(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms=None,mode='Training'):
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode

        # Load all image file names, sorting them to ensure correct matching with annotations
        if mode == 'Training':
            self.gt_path = os.path.join(self.data_path,'points','train.pkl')
            with open(self.gt_path,'rb') as f:
                data = pickle.load(f)
            self.name_list = list(data.keys())
        elif mode == 'Validation':
            self.gt_path = os.path.join(self.data_path,'points','val.pkl')
            with open(self.gt_path,'rb') as f:
                data = pickle.load(f)
            self.name_list = list(data.keys())
        elif mode == 'Testing':
            self.gt_path = os.path.join(self.data_path,'points','test.pkl')
            with open(self.gt_path,'rb') as f:
                data = pickle.load(f)
            self.name_list = list(data.keys())


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # Load images and annotations
        id = self.name_list[idx].split('_')[0]

        img_path = os.path.join(self.data_path, 'imgs', id+'.jpg')
        with open(self.gt_path,'rb') as f:

            gt_point_dict = pickle.load(f)
            point = gt_point_dict[self.name_list[idx]]
        
        img = Image.open(img_path).convert('RGB')

        # Extract bounding boxes, labels, and masks from the gts
        level = self.name_list[idx].split('_')[1]

        if 'BG' not in level:
            if level == 'C2':
                label = 3
            elif level == 'S1':
                label = 2
            else:
                label = 1
        else:
            label = 0
            background_sample_area = np.load(os.path.join(self.data_path,'resnet_background_sample_areas',id+'.npy'))
            point = random_click(background_sample_area)

        patch = generate_patch(img,center=point,patch_size=512)
         
        # Convert to tensors
        label = torch.as_tensor(np.array(label), dtype=torch.int64)

        # Target dictionary containing all the necessary components
        target = {}
        target['labels'] = label
        target['image_id'] = self.name_list[idx]

        if self.transforms:
            patch=self.transforms(patch)
        
        patch = torch.nn.functional.interpolate(patch.unsqueeze(0),size=(224,224)).squeeze(0)

        return patch, target

def collate_fn(batch):
    inputs,targets = zip(*batch)
    return tuple(inputs),tuple(targets)

class NHANESII(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms=None,mode='Training'):
        self.data_path = data_path
        self.transforms = transforms

        # Load all image file names, sorting them to ensure correct matching with annotations
        if mode == 'Training':
            with open(os.path.join(self.data_path,'..','data_split','train.txt'),'r') as f:
                self.name_list = [line[0:6] for line in f.readlines()]
        elif mode == 'Validation':
            with open(os.path.join(self.data_path,'..','data_split','val.txt'),'r') as f:
                self.name_list = [line[0:6] for line in f.readlines()]
        elif mode == 'Testing':
            with open(os.path.join(self.data_path,'..','data_split','test.txt'),'r') as f:
                self.name_list = [line[0:6] for line in f.readlines()]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # Load images and annotations
        img_path = os.path.join(self.data_path, 'imgs', self.name_list[idx]+'.jpg')
        gt_paths = [file for file in os.listdir(os.path.join(self.data_path, 'gts')) if file.split('_')[0] == self.name_list[idx]]
        img = Image.open(img_path).convert('RGB')

        # Extract bounding boxes, labels, and masks from the gts
        boxes = []
        labels = []
        masks = []
        centroids = []

        for gt_path in gt_paths:
            gt = np.load(os.path.join(self.data_path,'gts',gt_path)).astype(np.uint8)
            box = regionprops(gt)[0]['bbox']
            c_y,c_x = regionprops(gt)[0]['centroid']
            box = (box[1],box[0],box[3],box[2])
            level = gt_path.split('_')[1].split('.')[0]
            label = 1
            
            masks.append(gt)
            labels.append(label)
            boxes.append(box)
            centroids.append((c_x,c_y))
            
        # Convert to tensors
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.int16)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        # Target dictionary containing all the necessary components
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['centroids'] = centroids
        target['image_id'] = self.name_list[idx]

        if self.transforms:
            img= self.transforms(img)

        return img, target

def collate_fn(batch):
    inputs,targets = zip(*batch)
    return tuple(inputs),tuple(targets)



