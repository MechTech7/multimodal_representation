#from dataloaders import MultiModalManipulationDataset
from torch.utils import data
from dataloaders.TactoManipulationDataset import TactoManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import numpy as np
from models.tacto_base_models.encoders import TactoColorEncoder, TactoDepthEncoder, FullTactoColorEncoder, FullTactoDepthEncoder
import torch

def main():
    dataset = TactoManipulationDataset()
    sampler = SubsetRandomSampler(range(len(dataset)))
    device = torch.device("cuda")

    loader = DataLoader(dataset, batch_size=64, 
                num_workers=0, sampler=sampler, drop_last=True)

    print (f"d_keys: {dataset[0].keys()}")
    # Opens a new window and displays the output frame
    #for i in range(len(dataset)):
    #    print (f"{dataset[i]['pre_action']['proprio']}")
    
    k = cv2.waitKey(20000)
    count = 0

    #tacto_c_enc = TactoColorEncoder(128).to(device)
    #tacto_d_enc = TactoDepthEncoder(128).to(device)

    full_c_enc = FullTactoColorEncoder(128).to(device)
    full_d_enc = FullTactoDepthEncoder(128).to(device)

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

        clr_imgs = sample_batched['digits_color'].to(device)

        first_depth = sample_batched['digits_depth'].to(device)

        op_clr = full_c_enc(clr_imgs)
        #print(f"sing_img_shape: {}")
        op_depth = full_d_enc(first_depth)

        print (f"color_shape: {op_clr.shape}")
        print (f"depth_shape: {op_depth.shape}")
        count += 1
    
    #print (f"dataset[100]: {dataset[100]}")

if __name__ == "__main__":
    main()