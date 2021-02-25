from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from glob import glob
from utils import get_id_from_filename
import bisect
import torch
from math import floor


class ROI_dataset(Dataset):
    """
    loads main_modality and any other specified data and concatenates them. also loads the labels.
    * scans will be loaded based on ids available in main_modality_folder_path.
    """

    def __init__(self, main_modality_folder_path, inputs_folder_names, label_folder_path, output_label_img=False):
        """
        Initialization
        :param main_modality_folder_path: path to the folder containing input main_modality scans. should
                                        be named 'main_modality', and contain nifti (.nii.gz) images.
                                        images are named: 'BB[number id].nii.gz'
        :param inputs_folder_names: list of folder names of inputs in addition to main_modality.
                                    * only folder name should be different in path (replaced with 'main_modality' in path)
        :param label_folder_path: path to the folder containing input labels (nifti files).
                                    labels are named: 'label_nifti[number id].nii.gz'
        :param output_label_img: if it will output the dicom format of label as well as npy (for dice calculation)
        """
        self.main_modality_folder_path = main_modality_folder_path
        self.label_folder_path = label_folder_path

        self.other_input_folder_paths = [main_modality_folder_path.replace('main_modality', x) for x in inputs_folder_names]
        self.output_label_img = output_label_img
        self.data_list = sorted(glob(main_modality_folder_path + '/*'))

    def __len__(self):
        # returns the total number of samples
        return len(self.data_list)

    def __getitem__(self, idx):
        scan_id = get_id_from_filename(self.data_list[idx], all_numbers=True)
        # loading data and label and returning them
        main_modality_path = self.main_modality_folder_path + '/BB' + scan_id + '.nii.gz'
        label_path = self.label_folder_path + '/label_nifti' + scan_id + '.nii.gz'

        main_modality = sitk.ReadImage(main_modality_path)
        main_modality = sitk.GetArrayFromImage(main_modality)
        main_modality = main_modality.astype(np.float32)
        main_modality = main_modality[None, :, :, :]  # so data has shape [channel, x, y, z]

        label_img = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_img)
        label = label.astype(np.float32)
        label = label[None, :, :, :]

        # loading other input channels and concatenating to main_modality
        for input_folder in self.other_input_folder_paths:
            scan_path = input_folder + '/BB' + scan_id + '.nii.gz'
            scan = sitk.ReadImage(scan_path)
            scan = sitk.GetArrayFromImage(scan)
            scan = scan.astype(np.float32)
            scan = scan[None, :, :, :]

            main_modality = np.concatenate((main_modality, scan), axis=0)

        _data = dict()
        _data['data'] = main_modality
        _data['label'] = label
        if self.output_label_img:
            _data['label_img_path'] = label_path

        return _data


class ROI_collate(object):
    def __init__(self, acceptable_batch_sizes):
        self.acceptable_batch_sizes = acceptable_batch_sizes

    def __call__(self, output_dict_list):
        """
        Collate_fn takes the output of the dataloader __getitem__ and creates a batch. This
        is a costume collate_fn to create batches of data with different sizes by padding them
        so the final batch size is a power of two.
        * input data tensors in the list are of shape: [channel, x, y, z], where channel is equal for all data points.
        :param output_dict_list:
        :return:
        """

        # look at data and finds the closest acceptable batch size to the biggest element in data.
        acceptable_batch_sizes = self.acceptable_batch_sizes

        def take_first_bigger(ref_list, threshold):
            # returns the first biggest number after threshold in the ref_list.
            # ref_list should be sorted.
            return ref_list[bisect.bisect_left(ref_list, threshold)]

        data_list = [i['data'] for i in output_dict_list]
        label_list = [i['label'] for i in output_dict_list]

        X = take_first_bigger(ref_list=acceptable_batch_sizes,
                              threshold=max([i.shape[-3] for i in data_list]))
        Y = take_first_bigger(ref_list=acceptable_batch_sizes,
                              threshold=max([i.shape[-2] for i in data_list]))
        Z = take_first_bigger(ref_list=acceptable_batch_sizes,
                              threshold=max([i.shape[-1] for i in data_list]))

        # pad the data in the batch to the desired batch_size (X,Y,Z)
        def pad_batch(sequence, X, Y, Z):
            """
            :param sequence: list containing tensors
            :params X, Y, Z: desired batch size
            :return: batch tensor
            """
            channel_size = sequence[0].shape[-4]
            out_tensor = torch.zeros((len(sequence), channel_size, X, Y, Z))

            for i, tensor in enumerate(sequence):
                tensor = torch.Tensor(tensor)
                # tensor size
                Tx, Ty, Tz = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
                # left padding size
                Px, Py, Pz = floor((X - Tx) / 2), floor((Y - Ty) / 2), floor((Z - Tz) / 2)

                out_tensor[i, :, Px:Px + Tx, Py:Py + Ty, Pz:Pz + Tz] = tensor
            return out_tensor

        batch_data = pad_batch(data_list, X, Y, Z)
        batch_label = pad_batch(label_list, X, Y, Z)

        # returns the dictionary again
        _out = dict()
        _out['data'] = batch_data
        _out['label'] = batch_label
        if 'label_img_path' in list(output_dict_list[0].keys()):
            id_list = [i['label_img_path'] for i in output_dict_list]
            return _out, id_list
        else:
            return _out
