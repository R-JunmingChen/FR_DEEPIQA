
import  numpy as np
import  torch
import torch.nn as nn
import torchvision
from dataset.tid2013 import Tid2013Dataset
from dataset.tid2013 import BASE_PATH
from model.model import DeepQANet




image_datasets = {x: Tid2013Dataset(BASE_PATH,None,x)
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=12)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model=DeepQANet().to(device)
model.eval()


for batch_idx, ( r_set,d_set,mos) in enumerate(dataloaders['train']):
    r_set.to(device),d_set.to(device),mos.to(device)
    out = model(r_set, d_set)
    a = 1






#train_model(model, dataloaders,dataset_sizes,device,criterion, optimizer, scheduler, num_epochs=25)



