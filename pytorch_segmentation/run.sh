#!/bin/bash
#python train.py -c configs/enet_cross.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/enet_cross_1000.txt
#python train.py -c configs/enet_focal.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/enet_focal_1000.txt
python train.py -c configs/pspnet_cross_resnet101.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/pspnet_cross_resnet101.txt
#python train.py -c configs/pspnet_focal.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/pspnet_focal_1000.txt
#python train.py -c configs/unet_cross.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/unet_cross_1000.txt
#python train.py -c configs/unet_focal.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/unet_focal_1000.txt
python train.py -c configs/upernet_cross_resnet101.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/upernet_cross_resnet101.txt
#python train.py -c configs/upernet_focal.json 2>&1 | tee -a /mnt/1058CF1419A58A26/saved/upernet_focal_1000.txt
