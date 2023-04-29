# cs614 assignment 1

## assignment
All of the assignment has been documented in the assignment1.pdf

if you would like to run the code, please see one of the 2 sections below `jupyter notebook`, `training the mode`, or `super resolution`

## setup
create a conda environement and install the dependancies from the requirements.txt

run the following command to configure the tensor flow environement for GPU usage (option)
`source setup_env.sh`

## jupyter notebook
open jupyter notebook assignment1.ipnb
**Note:** you may  need to install and configure ipykernel to get the conda dependancies into your jupyter notebook

## training the model
run the training script 
`./cifar10_gan.py`
**Note:** see the `-h` option for details on what options are available

## Super resolution 
since the generated images are of low resolutions I have included an addition script that users a tensor flow super resolution module to increase the resolutions of the images 
`./super_resolution.py <saved generator model>`
**Note:** see the `-h` option for details on what options are available

