from __future__ import print_function, division
import torch, os, glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2



class LabDataset(Dataset):

    def __init__(self, rootdir=None, filelist=None, resize=None):

        if filelist:
            self.file_list = filelist
        else:
            assert os.path.exists(rootdir), "@dir:'%s' NOT exist ..."%rootdir
            self.file_list = glob.glob(os.path.join(rootdir, '*.*'))
            self.file_list.sort()
        self.resize = resize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        bgr_img = cv2.imread(self.file_list[idx], cv2.IMREAD_COLOR)
        if self.resize:
            bgr_img = cv2.resize(bgr_img, (self.resize,self.resize), interpolation=cv2.INTER_CUBIC)
        bgr_img = np.array(bgr_img / 255., np.float32)
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        #print('--------L:', np.min(lab_img[:,:,0]), np.max(lab_img[:,:,0]))
        #print('--------ab:', np.min(lab_img[:,:,1:3]), np.max(lab_img[:,:,1:3]))
        lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
        bgr_img = torch.from_numpy(bgr_img.transpose((2, 0, 1)))
        gray_img = (lab_img[0:1,:,:]-50.) / 50.
        color_map = lab_img[1:3,:,:] / 110.
        bgr_img = bgr_img*2. - 1.
        return {'gray': gray_img, 'color': color_map, 'BGR': bgr_img}
    


class LabDatasetCustom(Dataset):

    def __init__(self, imgs_list=None, resize=None):

        self.imgs_list = imgs_list
        self.resize = resize

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img_color_dir = self.imgs_list[idx][0]
        img_gray_dir = self.imgs_list[idx][1]
        
        img_bgr = cv2.imread(img_color_dir, cv2.IMREAD_COLOR)
        img_gray = cv2.imread(img_gray_dir, cv2.IMREAD_COLOR)
        
        img_bgr = cv2.resize(img_bgr, (self.resize, self.resize), interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.resize(img_gray, (self.resize, self.resize), interpolation=cv2.INTER_CUBIC)
        
        img_bgr = np.array(img_bgr / 255., np.float32)
        img_bgr2lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        img_gray = np.array(img_bgr / 255., np.float32)
        img_gray2lab = cv2.cvtColor(img_gray, cv2.COLOR_BGR2LAB)
        
        img_bgr2lab = torch.from_numpy(img_bgr2lab.transpose((2, 0, 1)))
        img_gray2lab = torch.from_numpy(img_gray2lab.transpose((2, 0, 1)))
        img_bgr = torch.from_numpy(img_bgr.transpose((2, 0, 1)))
        img_gray = (img_gray2lab[0:1,:,:]-50.) / 50.
        img_color = img_bgr2lab[1:3,:,:] / 110.
        img_bgr = img_bgr*2. - 1.
        return {'gray': img_gray, 'color': img_color, 'BGR': img_bgr}
    
    
    
class LabDatasetCustom(Dataset):

    def __init__(self, imgs_list=None, resize=None):

        self.imgs_list = imgs_list
        self.resize = resize

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        img_color_dir = self.imgs_list[idx][0]
        img_gray_dir = self.imgs_list[idx][1]
        
        img_bgr = cv2.imread(img_color_dir, cv2.IMREAD_COLOR)
        img_gray = cv2.imread(img_gray_dir, cv2.IMREAD_COLOR)
        
        img_bgr = cv2.resize(img_bgr, (self.resize, self.resize), interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.resize(img_gray, (self.resize, self.resize), interpolation=cv2.INTER_CUBIC)
        
        img_bgr = np.array(img_bgr / 255., np.float32)
        img_bgr2lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        img_gray = np.array(img_bgr / 255., np.float32)
        img_gray2lab = cv2.cvtColor(img_gray, cv2.COLOR_BGR2LAB)
        
        img_bgr2lab = torch.from_numpy(img_bgr2lab.transpose((2, 0, 1)))
        img_gray2lab = torch.from_numpy(img_gray2lab.transpose((2, 0, 1)))
        img_bgr = torch.from_numpy(img_bgr.transpose((2, 0, 1)))
        img_gray = (img_gray2lab[0:1,:,:]-50.) / 50.
        img_color = img_bgr2lab[1:3,:,:] / 110.
        img_bgr = img_bgr*2. - 1.
        return {'gray': img_gray, 'color': img_color, 'BGR': img_bgr}