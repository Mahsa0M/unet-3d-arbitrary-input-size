import torch

import torch
from math import floor
import sys

ROOT_DIR = ''
sys.path.append(ROOT_DIR)


def set_device():
    # CUDA if availabel
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    return device


def un_pad(img, output_dims):
    """
    un_pad the img to the output_dims. assumes the img is in the center.
    :param img: torch tensor, the image that is subject to unpadding. should be 3D
    :param output_dims: tuple, desired dimensions
    :return: torch tensor, unpadded image
    """
    assert len(img.shape) == 3
    # img size
    Tx, Ty, Tz = img.shape[0], img.shape[1], img.shape[2]
    # left padding size
    X, Y, Z = output_dims[0], output_dims[1], output_dims[2]
    Px, Py, Pz = floor((Tx - X) / 2), floor((Ty - Y) / 2), floor((Tz - Z) / 2)

    return img[Px:X + Px, Py:Y + Py, Pz:Z + Pz]


def get_id_from_filename(filename, all_numbers=True):
    """
    get id from a file or folder name
    :param filename: file path or file name (or folder). id must be in the last section after '/', and
                    it should be the only number in the name.
    :param all_numbers: if False, the id MUST start wth '0'. Otherwise it can start with any number.
    :return: patient id (str)
    """
    file_id = filename.split('/')[-1]
    file_id = file_id.split('.')[0]

    if all_numbers == False:
        file_id = file_id[file_id.find('0'):]
    if all_numbers == True:
        def where(list, element):
            # find all ocurances of element in list
            indx = [i for i, val in enumerate(list) if val == element]
            return indx

        # find locations of all numbers in the name
        num_locations = where(file_id, '0') + where(file_id, '1') + where(file_id, '2') + \
                        where(file_id, '3') + where(file_id, '4') + where(file_id, '5') + \
                        where(file_id, '6') + where(file_id, '7') + where(file_id, '8') + \
                        where(file_id, '9')
        try:
            file_id = file_id[min(num_locations):max(num_locations) + 1]
        except:
            raise ValueError('File name did not have a number in it.')
    return file_id


def overwrite_or_create_dir(dir_path):
    """
    creates a new directory.
    If the directory already exists, deletes it and creates a new one again (overwrites it).
    :param dir_path:
    :return: the dir_path
    """
    import os
    import shutil
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)  # Removes all the subdirectories
        os.makedirs(dir_path)
    return dir_path
