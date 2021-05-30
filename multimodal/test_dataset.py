#from dataloaders import MultiModalManipulationDataset
from torch.utils import data
from dataloaders.TactoManipulationDataset import TactoManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import numpy as np

def main():
    dataset = TactoManipulationDataset()
    sampler = SubsetRandomSampler(range(len(dataset)))

    loader = DataLoader(dataset, batch_size=2, 
                num_workers=0, sampler=sampler, drop_last=True)

    print (f"d_keys: {dataset[0].keys()}")
    # Opens a new window and displays the output frame
    #for i in range(len(dataset)):
    #    print (f"{dataset[i]['pre_action']['proprio']}")
    
    k = cv2.waitKey(20000)
    count = 0
    """for i_batched, sample_batched in enumerate(loader):
        #print ("--------------------------")
        #print (f"Sample: {sample_batched['pre_action']['proprio']}")
        #print (i[0])

        count += 1"""
    
    #print (f"dataset[100]: {dataset[100]}")

if __name__ == "__main__":
    main()