#from dataloaders import MultiModalManipulationDataset
from torch.utils import data
from dataloaders.TactoManipulationDataset import TactoManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import numpy as np

def get_optical_flow(sample):
    first_image = sample["pre_action"]["cam_color"]
    sec_image = sample["post_action"]["cam_color"]

    mask = np.zeros_like(first_image)

    first_gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)
    sec_gray = cv2.cvtColor(sec_image, cv2.COLOR_RGB2GRAY)

    print (f"{first_gray.shape}")
    flow = cv2.calcOpticalFlowFarneback(first_gray, sec_gray, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    print (f"flow shape: {flow.shape}")
    mask[..., 1] = 255

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print (f"angle shape: {angle.shape}")

    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return rgb

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