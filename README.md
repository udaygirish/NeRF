# NeRF - Pytorch Implementation

## Current ToDo:

 
    - [x] Basic Dataloader - Load Images and Data 
    - [x] Random Rays Generator - DataLoader for Training
    - [x] Load everything from Config
    - [x] Directional encoding 
    - [x] NDC [-1, 1]
    - [x] Hierrachial Sampling- Corase and Fine
    - [x] Noise to the Density (Gaussian)
    - [No] Tensorboard - Instead Weights & Biases
    - [ ] Parallel Training - Multi GPU ?
    - [X] Visualization (Rendering)
    - [X] Load Checkpoint and Render
    - [ ] Optimize for CPU 
    - [ ] Serve as a Web App

## Sample Output 

### Lego with Positional Encoding

![LegoWithPositionalEncoding](./Lego_With_Positional_Encoding.gif.gif)

### Lego without Positional Encoding

![LegoWithPositionalEncoding](./Lego_Without_Positional_Encoding.gif)

### Ship with Positional Encoding

![ShipWithPositionalEncoding](./Ship_With_Positional_Encoding.gif)

## To run the Code

please execute the below

        python3 Wrapper.py

To change any variable please use the config.yaml 
For new configurations make copies and the script takes argument --config to use a custom config.yaml. Every parameter explanation is present in the Yaml under help.


This code logs parameters in Wandb, please login using weights and biases and install wandb from pip then execute wandb -login to get api key and log your results.

We have also added scripts for generation for real experiments videos in utilities <colmap2nerf.py> for generating the Transforms and <remove_bg.py> for removing backgrounds. 
You need Colmap locally installed for running colmap script. This script is directly taken from NVIDIA instant NGP repo.

### This code base is loosely based on three implementations
1. NERF Original Repo (https://github.com/bmild/nerf) - Written in Tensorflow by original authors
2. NERF With Pytorch (https://github.com/yenchenlin/nerf-pytorch/tree/master) - Pytorch (One of the best cited repos)
3. NERF Pytorch (Faster Version - https://github.com/krrish94/nerf-pytorch) - Faster implementation based on Repo 2.

### Multi GPU training can achieved by Repo 3 or with the below Repo
1. https://github.com/dogyoonlee/nerf-pytorch-multi-gpu/tree/main (Uses NN.parallel)
2. Implementing further you can also check instant-ngp by NVIDIA. (One of the fastest Repo's)


### For any queries please email : umaradana@wpi.edu, pshinde@wpi.edu