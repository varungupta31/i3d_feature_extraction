import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from cmd_dataloader import CondensedMoviesClips as Dataset
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)
args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

print("-------------------------------------")
#print args parse summary on the terminal
print("Mode: ", args.mode)
print("Load Model: ", args.load_model)
print("Root: ", args.root)
print("GPU: ", "cuda:"+args.gpu)
print("Save Dir: ", args.save_dir)
print("-------------------------------------")


if(args.gpu):
    device = torch.device("cuda:"+args.gpu)


def run(mode='rgb', root='/ssd_scratch/cvit/varun/videos', file='vid_info.json', batch_size=1, load_model='', save_dir=''):

    transform = transforms.Compose([videotransforms.RandomCrop(224)])
    dataset = Dataset(file, root, mode, transform, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=8, pin_memory = True)

    print("Loading I3D model ...")
    if mode == "rgb":
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(400)
    i3d.load_state_dict(torch.load(load_model))
    #i3d.cuda()
    i3d.to(device = device)

    i3d.train(False)

    for data in tqdm(dataloader, desc="Extracting Video Features and Saving to"+ args.save_dir):

        inputs, name = data
        if os.path.exists(os.path.join(save_dir, name[0]+".npy")): #TODO Check if name[0] is correct.
            continue

        b, c, t, h, w = inputs.shape
        if t > 1600:
            with torch.no_grad():
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).to(device = device), volatile=True)
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
        else:
            with torch.no_grad():
                # wrap them in Variable
                #inputs = Variable(inputs.cuda(), volatile=True)
                inputs = Variable(inputs.to(device=device))
                features = i3d.extract_features(inputs)
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())





if __name__ == '__main__':
    run(mode=args.mode, root=args.root, save_dir=args.save_dir, load_model=args.load_model)
    
    