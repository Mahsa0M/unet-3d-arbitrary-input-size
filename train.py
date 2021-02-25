import os
from datetime import date, datetime
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from validation import validate
from dataset import *
from models import *
from visualize import visualizer
from utils import *

bin_path = os.path.join(ROOT_DIR,'bin')
config_path = os.path.join(ROOT_DIR,'train_config.yaml')

# set device
device = set_device()
print('device = ', device)

# read config file
with open(config_path, "r") as yamlfile:
    cfg = yaml.load(yamlfile)

# Setting the seed
seed = cfg['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# initialize model
model = UNet3D_4layer(num_inputs=1 + len(cfg['additional_inputs_folder_names']))
model = model.to(device)

# initialize dataloader and collate function
# collate function creates batches. we need a costume one to pad batches to acceptable sizes.
collate_function = ROI_collate(acceptable_batch_sizes=model.acceptable_batch_sizes)

# initialializing datasets depending on model
train_dataset = ROI_dataset(main_modality_folder_path=cfg['train_main_modality_folder_path'],
                            inputs_folder_names=cfg['additional_inputs_folder_names'],
                            label_folder_path=cfg['train_label_folder_path'],
                            output_label_img=False)

val_dataset = ROI_dataset(main_modality_folder_path=cfg['val_main_modality_folder_path'],
                          inputs_folder_names=cfg['additional_inputs_folder_names'],
                          label_folder_path=cfg['val_label_folder_path'],
                          output_label_img=True)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_function)

val_dataloader = DataLoader(dataset=val_dataset,
                            # batch_size=cfg['batch_size'],
                            batch_size=1,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_function)

# initialize experiment name and folders
exp_label = model.model_name + '-main_modality'
for input in cfg['additional_inputs_folder_names']:
    exp_label += '-' + input

run_info = exp_label + '_' + date.today().strftime("_%b-%d-%Y") + datetime.now().strftime("_time%H-%M-%S")

model_folder = os.path.join(bin_path, 'models', run_info)
os.mkdir(model_folder)  # for saving the models

# for visualization
if cfg['val_log_every'] == 'epoch':
    cfg['val_log_every'] = len(train_dataloader) - 1

vis_obj = visualizer(cfg=cfg,
                     log_dir=os.path.join(bin_path, 'visualize', run_info))

# initialize optimizer and loss
if cfg['loss_pos_weight'] != None:
    pos_weight = torch.Tensor([cfg['loss_pos_weight']]).float().to(device)
else:
    pos_weight = None
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=cfg['optimizer_lr'])

# main training loop
print('Number of batches: ', len(train_dataloader))

running_loss = 0

for epoch_num, epoch in enumerate(range(cfg['max_epochs'])):
    for step, _data in enumerate(train_dataloader):

        model.train()

        # sending to device
        data, label = _data['data'].to(device), _data['label'].to(device)
        del _data

        # training step
        optimizer.zero_grad()
        pred = model.forward(data)

        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        # update visualization vars
        running_loss += loss.detach().item()

        # log train loss
        if step % cfg['train_log_every'] == 0:
            vis_obj.log_train({'running_loss': running_loss})

        # validation
        if step % cfg['val_log_every'] == 0 and step != 0:
            print('Validating...')
            _, _ = validate(model, val_dataloader, criterion, vis_obj)
            del _

    # save model
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimizer.state_dict()},
               model_folder + '/E' + str(epoch_num) + '.tar')

    print('Epoch ', epoch_num, ' finished.')
