## Edges-to-Shoes experiment

First you need to prepare data as explained in `datasets` folder

### Training Augmented CycleGAN model
`CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/font2hand/ --name augcgan_model`