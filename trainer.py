import  torch
import time
from dataset.live import Tid2013Dataset
from dataset.live import BASE_PATH
from model.model import DeepQANet
from scipy.stats import spearmanr, pearsonr, kendalltau
import copy


#@todo 1. config system 2. logging system

MODEL_SAVE_PATH='/Users/mayzha/PycharmProjects/model_st.pt'

def train_model(model, dataloaders,device, optimizer, num_epochs=25):

    since = time.time()
    best_PLCC = 0.0
    best_SRCC = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            groundtruth_mos_set=[]
            predict_mos_set=[]
            epoch_phase_loss = 0.0
            epoch_phase_size=0

            # Iterate over data.
            for batch_index,(r_patch_set,d_patch_set, mos_set) in enumerate(dataloaders[phase]):
                r_patch_set = r_patch_set.to(device)
                d_patch_set = d_patch_set.to(device)
                mos_set = mos_set.to(device)

                #reshape
                r_patch_set=r_patch_set.reshape(r_patch_set.shape[1],r_patch_set.shape[2],r_patch_set.shape[3],r_patch_set.shape[4])
                d_patch_set=d_patch_set.reshape(d_patch_set.shape[1], d_patch_set.shape[2], d_patch_set.shape[3],
                                    d_patch_set.shape[4])



                # zero the parameter gradients
                optimizer.zero_grad()




                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    total_loss,predict_mos = model(r_patch_set,d_patch_set,mos_set)
                    predict_mos=predict_mos.reshape(mos_set.shape)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # statistics
                current_loss =total_loss.item() * r_patch_set.size(0)
                epoch_phase_loss += current_loss
                epoch_phase_size += r_patch_set.size(0)


                groundtruth_mos_set.append(mos_set.flatten())
                predict_mos_set.append(predict_mos.flatten())




                #print('batch {} Loss: {:.4f} '.format(batch_index, current_loss))

            epoch_averge_loss = epoch_phase_loss / epoch_phase_size

            groundtruth_mos_set = torch.cat(groundtruth_mos_set).flatten().data.numpy()
            predict_mos_set = torch.cat(predict_mos_set).flatten().data.numpy()

            epoch_PLCC = pearsonr(groundtruth_mos_set,predict_mos_set)[0] #(corr,p value)
            epoch_SRCC = spearmanr(groundtruth_mos_set, predict_mos_set)[0]#(corr,p value)

            print('{} Loss: {:.4f} PLCC: {:.4f} SRCC: {:.4f}'.format(
                phase, epoch_averge_loss, epoch_PLCC,epoch_SRCC))

            # deep copy the model
            #@todo to save what
            if phase == 'test' and (epoch_PLCC > best_PLCC  or epoch_SRCC > best_SRCC):
                best_PLCC = epoch_PLCC
                best_SRCC = epoch_SRCC

                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, MODEL_SAVE_PATH)# to save model_st_dic in disk



        print('Epoch {}/{} done'.format(epoch, num_epochs - 1))
        print('-' * 10)



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best  PLCC: {:4f}  Best  SRCC: {:4f}'.format(best_PLCC,best_SRCC))

    model.load_state_dict(best_model_wts)

    return model

if __name__=='__main__':
    image_datasets = {x: Tid2013Dataset(BASE_PATH, None, x)
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'test']}
    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('use {0}'.format( 'cuda'if torch.cuda.is_available() else  'cpu'))

    model = DeepQANet(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_model_wts=train_model(model, dataloaders,device, optimizer, num_epochs=100)

