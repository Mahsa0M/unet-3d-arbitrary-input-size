from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk

class ROI_dataset(Dataset):
    def __init__(self, id_list_file, data_folder_path, label_folder_path, output_label_img = False):
        """
        Initialization
        :param id_list_file: list of patient ids of input images (txt file)
        :param data_folder_path: path to the folder containing input scans (DICOM)
        :param label_folder_path: path to the folder containing input labels (DICOM)
        :param output_label_img: if it will output the dicom format of label as well as npy (for dice calculation)
        """
        self.data_folder_path = data_folder_path
        self.label_folder_path = label_folder_path
        self. output_label_img = output_label_img
        with open(id_list_file, 'r') as f:
            self.id_list = f.readlines()

    def __len__(self):
        # returns the total number of samples
        return len(self.id_list)

    def __getitem__(self, idx):
        # loading data and label and returning them
        data_path = self.data_folder_path + '/ROI' + str(idx) + '.nii.gz'
        label_path = self.label_folder_path + '/label' + str(idx) + '.npy'

        data = sitk.ReadImage(data_path)
        data = sitk.GetArrayFromImage(data)
        label_img = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_img)

        _data = {}
        _data['data'] = data
        _data['label'] = label
        if self.output_label_img:
            _data['label_img'] = label_img

        return _data

