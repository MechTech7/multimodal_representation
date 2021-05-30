import h5py
import numpy as np
import ipdb
from tqdm import tqdm
from dataloaders.util import DatasetUtils
from torch.utils.data import Dataset, dataset

class TactoManipulationDataset(Dataset):
    def __init__(self,
                 transform=None, 
                 dataset_location="/home/mason/peg_insertation_dataset/heuristic_data_contact_1/") -> None:
        self.dataset_location = dataset_location
        self.data_util = DatasetUtils(dataset_location)
        self.transform = transform
        self.pairing_tolerance = 0.2 #placeholder value

        #NOTE: I know it's not the best idea to load everything into RAM but it seems to work for now
        self.data_list = self.data_util.recall_obs()
        self._init_paired_filenames()

        pass

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        unpaired_idx = self.paired_examples[index]
        example = self._get_single(index, unpaired_idx)

        return example#self.data_list[index]

    def _get_single(self, index, unpaired_idx):
        example = self.data_list[index]
        unpaired_ex = self.data_list[unpaired_idx]

        image = example["cam_color"]
        depth = example["cam_depth"]
        digits_color = example["digits_color"]
        digits_depth = example["digits_depth"]
        proprio = example["proprio"][:8]

        flow = example["optical_flow"]

        flow_mask = np.expand_dims(
            np.where(
                flow.sum(axis=2) == 0,
                np.zeros_like(flow.sum(axis=2)),
                np.ones_like(flow.sum(axis=2)),
            ),
            2,
        )

        unpaired_proprio = unpaired_ex["proprio"][:8]
        unpaired_digits_color = unpaired_ex["digits_color"]
        unpaired_digits_depth = unpaired_ex["digits_depth"]


        sample = {
            "image": image,
            "depth": depth,
            "flow": flow,
            "flow_mask": flow_mask,
            "action": self.data_list[index]["action"],
            "digits_color": np.array(digits_color),
            "digits_depth": np.array(digits_depth),
            "proprio": proprio,
            "ee_yaw_next": self.data_list[index]["ee_yaw_next"],
            "contact_next": np.array([self.data_list[index]["contact"]]).astype(np.float),
            "unpaired_image": image,
            "unpaired_depth": depth,
            "unpaired_proprio": unpaired_proprio,
            "unpaired_digits_color": unpaired_digits_color,
            "unpaired_digits_depth": unpaired_digits_depth
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _init_paired_filenames(self):
        tolerance = self.pairing_tolerance

        self.paired_examples = {}

        for idx in tqdm(range(self.__len__()), desc="pairing filnames"):
            proprio_dist = None
            curr_proprio = self.data_list[idx]["proprio"]
            while proprio_dist is None or proprio_dist < tolerance:
                #print (f"proprio_dist: {proprio_dist}")
                unpaired_idx = np.random.randint(self.__len__())

                while unpaired_idx == idx:
                    unpaired_idx = np.random.randint(self.__len__())
                
                unpaired_proprio = self.data_list[unpaired_idx]["proprio"]
            
                #print ("--------------------------")
                #print (f"Curr_proprio: {curr_proprio}")
                #print (f"unpaired_proprio: {unpaired_proprio}")
                proprio_dist = np.linalg.norm(np.array(curr_proprio[:3]) - np.array(unpaired_proprio[:3]))

                #print (f"paired_idx: {idx}")
                #print (f"unpaired_idx: {unpaired_idx}")
                
                
            self.paired_examples[idx] = unpaired_idx
        pass