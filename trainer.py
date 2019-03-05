
import  numpy as np
import  torch
import torch.nn as nn
import torchvision
import time
from dataset.live import Tid2013Dataset
from dataset.live import BASE_PATH
from model.model import DeepQANet
import copy


image_datasets = {x: Tid2013Dataset(BASE_PATH,None,x)
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=DeepQANet()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005,weight_decay=1e-4) #@todo l2 regulation


def spearman_correlation(x,y):

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    spearman_correlation_result= torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return  spearman_correlation_result


def train_model(model, dataloaders,dataset_sizes,device, optimizer, num_epochs=25):

    since = time.time()
    best_acc = 0.0
    mse_loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            mos_set=[]
            predict_mos_set=[]
            running_loss = 0.0

            # Iterate over data.
            for batch_index,(r_patch_set,d_patch_set, mos_set) in enumerate(dataloaders[phase]):
                r_patch_set = r_patch_set.to(device)
                d_patch_set = d_patch_set.to(device)

                r_patch_set=r_patch_set.reshape(r_patch_set.shape[1],r_patch_set.shape[2],r_patch_set.shape[3],r_patch_set.shape[4])
                d_patch_set=d_patch_set.reshape(d_patch_set.shape[1], d_patch_set.shape[2], d_patch_set.shape[3],
                                    d_patch_set.shape[4])

                mos_set = mos_set.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()




                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    predict_mos,tv_nomal_loss = model(r_patch_set,d_patch_set,mos_set)
                    predict_mos, tv_nomal_loss=predict_mos.flatten(),tv_nomal_loss.flatten()





                    total_loss=1
                    total_loss = total_loss.sum()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # statistics
                current_loss =total_loss.item() * r_patch_set.size(0)
                running_loss += current_loss
                mos_set.append(mos.flatten())
                predict_mos_set.append(predict_mos.flatten())

                print('batch {} Loss: {:.4f} '.format(
                    batch_index, current_loss))



            epoch_loss = running_loss / dataset_sizes[phase]
            mos_set=torch.cat(mos_set,dim=0)
            predict_mos_set=torch.cat(predict_mos_set,dim=0)
            epoch_acc = spearman_correlation(mos_set,predict_mos_set)



            print('{} Loss: {:.4f} SPCC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best  SPCC: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model

train_model(model, dataloaders,dataset_sizes,device, optimizer, num_epochs=35)