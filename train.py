import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from validation import validate
from dataset import ROI_dataset
from models import FC_3D_UNet

#TODO: debug
#TODO: find the min size for the input ROI, and a max size for memory limits??
#TODO: add visualiztion in validation ?

# set device
device = set_device()

# read config file
config_path = "config.yaml"
with open(config_path, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)

# initialize dataloader
train_dataset = ROI_dataset(id_list_file=cfg['train_id_list_file'],
                      data_folder_path=cfg['train_data_folder_path'],
                      label_folder_path=cfg['train_label_folder_path'],
                        output_label_img=False)

val_dataset = ROI_dataset(id_list_file=cfg['val_id_list_file'],
                      data_folder_path=cfg['val_data_folder_path'],
                      label_folder_path=cfg['val_label_folder_path'],
                        output_label_img=True)

train_dataloader = DataLoader(dataset=train_dataset,
                        batch_size=cfg['batch_size'],
                        shuffle=True,
                        num_workers=4)

val_dataloader = DataLoader(dataset=val_dataset,
                        batch_size=cfg['batch_size'],
                        shuffle=True,
                        num_workers=4)

# initialize for visualization
writer = SummaryWriter()
running_loss = 0

# initialize model
model = FC_3D_UNet()
model = model.to(device)

# initialize optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['optimizer_lr'])

# main training loop
for epoch in range(cfg['max_epochs']):
    for step, _data in enumerate(train_dataloader):
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
        running_loss += loss.item()

        # log train loss
        if step % cfg['train_log_every'] == 0:
            writer.add_scalar('train/loss', running_loss/cfg['train_log_every'], step)
            running_loss = 0

        # validation
        if step%cfg['val_log_every'] == 0:
            val_loss, val_dice = validate(model, val_dataloader, criterion)
            writer.add_scalar('validation/loss', val_loss, step)
            writer.add_scalar('validation/dice', val_dice, step)
            del val_loss, val_dice