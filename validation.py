from utils import *
import yaml
import os
from dataset import *
from torch.utils.data import DataLoader
from models import *
from visualize import visualizer
from datetime import date, datetime


def validate(model, val_dataloader, loss_criterion, vis_obj=None):
    device = set_device()

    model.eval()

    # initializing evaluation values
    running_loss = 0
    running_dice = 0
    num_samples = 0

    for _data, id_list in val_dataloader:
        # sending to device
        data, label = _data['data'].to(device), _data['label'].to(device)

        # outputting prediction
        pred = model.forward(data)

        # loss
        loss = loss_criterion(pred, label)
        running_loss += loss.item()

        # dice (applied over each image in batch separately)
        for i, id in enumerate(id_list):
            label_img = sitk.ReadImage(id)
            label_np = sitk.GetArrayFromImage(label_img)
            this_pred_pad = pred[i, :, :, :].squeeze()

            this_pred_pad_sig = torch.sigmoid(this_pred_pad)    # for visualization

            # unpad the predictions
            this_pred = un_pad(this_pred_pad, label_np.shape)
            this_pred = this_pred.detach().cpu().numpy()

            pred_img = sitk.GetImageFromArray(this_pred)
            pred_img = sitk.Cast(pred_img, label_img.GetPixelIDValue())
            pred_img.CopyInformation(label_img)

            # threshold using sitk thresholding. you can try different options.
            thresh_method = 'Otsu'
            threshold_filters = {'Otsu': sitk.OtsuThresholdImageFilter(),
                                 'Triangle': sitk.TriangleThresholdImageFilter(),
                                 'Huang': sitk.HuangThresholdImageFilter(),
                                 'MaxEntropy': sitk.MaximumEntropyThresholdImageFilter()}
            thresh_filter = threshold_filters[thresh_method]

            thresh_filter.SetInsideValue(0)
            thresh_filter.SetOutsideValue(1)
            pred_img = thresh_filter.Execute(pred_img)

            pred_img = sitk.Cast(pred_img, sitk.sitkUInt8)
            label_img = sitk.Cast(label_img, sitk.sitkUInt8)

            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_measures_filter.Execute(label_img, pred_img)

            dice = overlap_measures_filter.GetDiceCoefficient()
            running_dice += dice
            num_samples += 1

    avg_loss = running_loss / num_samples
    avg_dice = running_dice / num_samples

    # visualize
    if vis_obj != None:
        vis_obj.log_val({'val_loss': avg_loss,
                         'val_dice': avg_dice,
                         'binary_pred': torch.tensor(sitk.GetArrayFromImage(pred_img)).float(),
                         'label': label[-1, :, :, :].squeeze(),
                         'pred': this_pred_pad_sig})

    model.train()

    return avg_loss, avg_dice


if __name__ == '__main__':
    # for running validation seperately
    visualize = True
    load_model = False
    saved_model_path = ''

    # set device
    device = set_device()

    # read config file
    config_path = "train_config.yaml"
    with open(config_path, "r") as yamlfile:
        cfg = yaml.load(yamlfile)

    # in case visualize, initialize vis_obj
    if visualize:
        run_info = 'validation' + date.today().strftime("_%b-%d-%Y") + datetime.now().strftime("_time%H-%M-%S")
        vis_obj = visualizer(cfg=cfg,
                             log_dir=os.getcwd() + "/bin/visualize/" + run_info)
    else:
        vis_obj = None

    # initialize model
    model = UNet3D_4layer(num_inputs=1 + len(cfg['additional_inputs_folder_names']))
    model = model.to(device)

    if load_model:
        model_checkpoint = torch.load(saved_model_path)
        model.load_state_dict(model_checkpoint['model_state_dict'])

    # initialize dataloader
    collate_function = ROI_collate(acceptable_batch_sizes=model.acceptable_batch_sizes)

    val_dataset = ROI_dataset(main_modality_folder_path=cfg['val_main_modality_folder_path'],
                              inputs_folder_names=cfg['additional_inputs_folder_names'],
                              label_folder_path=cfg['val_label_folder_path'],
                              output_label_img=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                # batch_size=cfg['batch_size'],
                                batch_size=1,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=collate_function)

    # initialize criterion
    criterion = nn.BCEWithLogitsLoss()

    # validation
    val_loss, val_dice = validate(model, val_dataloader, criterion, vis_obj=vis_obj)
    print(val_loss, val_dice)
