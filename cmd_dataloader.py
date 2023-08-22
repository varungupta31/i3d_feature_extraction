import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import csv
import h5py
import os
import os.path
import cv2
import glob

def make_dataset(file, root):
    dataset = []
    with open(file, 'r') as f:
        data = json.load(f)
    
    i = 0
    for vid in data.keys():
        #check if the video directory exists
        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = int(data[vid])
        
        dataset.append((vid, num_frames))
        i+=1
    return dataset

def load_rgb_frames(root, vid, start, num):
    frames = []
    for i in range(start, start+num):
        img = cv2.imread(os.path.join(root, vid, 'frame_{:04d}.jpg'.format(i)))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def video_to_tensor(frame):
    return torch.from_numpy(frame.transpose([3, 0, 1, 2]))
    

class CondensedMoviesClips(Dataset):

    def __init__(self, file, root, mode, transforms=None, save_dir="", num=0):

        self.data = make_dataset(file, root)
        self.file = file
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.save_dir = save_dir


    def __getitem__(self, index):
        #Root contains all the folders with the video names.
        #Each folder contains the frames of the video.
        
        vid, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid + '.npy')):
            return 0, 0, vid
        
        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 1, nf)
        
        imgs = self.transforms(imgs)
        return video_to_tensor(imgs), vid
    
    def __len__(self):
        return len(self.data)