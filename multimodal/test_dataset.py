#from dataloaders import MultiModalManipulationDataset
from torch.utils import data
from dataloaders.TactoManipulationDataset import TactoManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import numpy as np
from models.tacto_base_models.encoders import (
        TactoColorEncoder, 
        TactoDepthEncoder, 
        FullTactoColorEncoder, 
        FullTactoDepthEncoder, 
        TactoEncoder, 
        PerlsImageEncoder, 
        PerlsDepthEncoder
)

from models.tacto_sensor_fusion import (
    SensorFusion,
    SensorFusionSelfSupervised,
)

from models.models_utils import rescaleImage
import torch

def main():
    dataset = TactoManipulationDataset()
    sampler = SubsetRandomSampler(range(len(dataset)))
    device = torch.device("cuda")

    loader = DataLoader(dataset, batch_size=64, 
                num_workers=0, sampler=sampler, drop_last=True)

   
    count = 0

    #tacto_c_enc = TactoColorEncoder(128).to(device)
    #tacto_d_enc = TactoDepthEncoder(128).to(device)

    full_c_enc = FullTactoColorEncoder(128).to(device)
    full_d_enc = FullTactoDepthEncoder(128).to(device)

    image_enc = PerlsImageEncoder(128).to(device)
    depth_enc = PerlsDepthEncoder(128).to(device)
    tacto_enc = TactoEncoder(128).to(device)

    sens_fuse = SensorFusion(device, z_dim=128, action_dim=3).to(device)
    sup_sens_fuse = SensorFusionSelfSupervised(device, z_dim=128, action_dim=3).to(device)


    for i_batched, sample_batched in enumerate(loader):
        #print ("--------------------------")
        #print (f"Sample: {sample_batched['pre_action']['proprio']}")
        #print (i[0])
        #print (f"image_shape: {sample_batched['image'].shape}")
        #print (f"depth_shape: {sample_batched['depth'].shape}")
        #print (f"ee_yaw_next_shape: {sample_batched['ee_yaw_next'].shape}")
        #print (f"proprio_shape: {sample_batched['proprio'].shape}")
        #print (f"contact_next: {sample_batched['contact_next'].shape}")
        #print (f"digits_color_shape: {sample_batched['digits_color'].shape}")
        #print (f"digits_depth_shape: {sample_batched['digits_depth'].shape}")

        clr_images = sample_batched['image'].to(device)
        #clr_images = rescaleImage(clr_images).to(device)
        
        depth_img = sample_batched['depth'].transpose(1, 3).transpose(2, 3).to(device)

        #cl_op, cl_convs = image_enc(clr_images)
        #dp_op, dp_convs = depth_enc(depth_img)

        tacto_clr_imgs = sample_batched['digits_color'].to(device)
        tacto_depth_imgs = sample_batched['digits_depth'].to(device)

        proprio = sample_batched['proprio'].to(device)
        action = sample_batched['action'].to(device)

        #fused = sens_fuse.forward_encoder(clr_images, tacto_clr_imgs, tacto_depth_imgs, proprio, depth_img, action)
        sup_fused = sup_sens_fuse(clr_images, tacto_clr_imgs, tacto_depth_imgs, proprio, depth_img, action)
        #op_clr = full_c_enc(clr_imgs)
        #print(f"sing_img_shape: {}")
        #op_depth = full_d_enc(first_depth)

        
        #tacto_lin_out = tacto_enc(tacto_clr_imgs, first_depth)
        #print (f"color_shape: {tacto_lin_out.shape}")
        #print (f"depth_shape: {op_depth.shape}")
        #print (f"img_color_shape: {dp_op.shape}")
        count += 1
    
    #print (f"dataset[100]: {dataset[100]}")

if __name__ == "__main__":
    main()