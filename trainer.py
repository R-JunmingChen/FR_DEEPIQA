
import  numpy as np
import  torch
import torch.nn as nn
import torchvision
from dataset.tid2013 import Tid2013Dataset
from dataset.tid2013 import BASE_PATH
from model.model import DeepQANet


image_datasets = {x: Tid2013Dataset(BASE_PATH,None,x)
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=DeepQANet()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005,weight_decay=1e-5) #@todo l2 regulation



mse_loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)




#train_model(model, dataloaders,dataset_sizes,device,criterion, optimizer, num_epochs=25)


for r_set,d_set,mos in  dataloaders['train']:
    model(r_set,d_set)



