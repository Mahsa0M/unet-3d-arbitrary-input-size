import SimpleITK as sitk
from utils import *

def validate(model, val_dataloader, loss_criterion):
    device = set_device()

    model.eval()

    # initializing evaluation values
    running_loss = 0
    running_dice = 0

    for _data in val_dataloader:
        # sending to device
        data, label, label_img = _data['data'].to(device), _data['label'].to(device), _data['label_img'].to(device)

        # outputting prediction
        pred = model.forward(data)

        # loss
        loss = loss_criterion(pred, label)
        running_loss += loss.item()

        # dice
        pred_img = sitk.GetImageFromArray(pred)
        pred_img = sitk.Cast(pred_img, label_img.GetPixelIDValue())
        pred_img.CopyInformation(label_img)
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(label_img, pred_img)

        dice = overlap_measures_filter.GetDiceCoefficient()
        running_dice += dice

    avg_loss = running_loss / len(val_dataloader)
    avg_dice = running_dice / len(val_dataloader)

    return avg_loss, avg_dice