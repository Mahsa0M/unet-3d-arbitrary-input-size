# 3D U-Net that accepts inputs of different sizes

Dataset is structured as: ([ ] means arbitrary name, ' ' means should be named exactly the same.)
 
        [dataset folder] ---- 'main_modality' --- data_train -- files named 'BB[number id].nii.gz'
                           |                   |
                           |                   -- data_val  -- files named 'BB[number id].nii.gz'
                           |
                           |-- label --- label_train -- files named 'label_nifti[number id].nii.gz'
                           |           |
                           |           -- label_val  -- files named 'label_nifti[number id].nii.gz'
                           |
                           -- [secondary input folder] --- ... similar to the 'main modality' folder
A dummy_dataset with inputs of different sizes is added.

* having more than one input is optional. The network does not have a limit on the number of input channels it accepts.

- Paths to dataset should be set in train_config.yaml.
    * In case of having more than one input channel, set the name of the other input folders in 'additional_inputs_folder_names'.

- Network has a number of acceptable input sizes, and padds the rest of the input sizes with zeros.
    * The list of acceptable input sizes is inside the model.
    * The lower limit is a limit of the model, but the upper limit can increase (you can find new acceptable sizes by trial and error!)

- You can try different thresholding options for dice calculation in validation, thresh_method.

- TO TRAIN: run 'train.py'
