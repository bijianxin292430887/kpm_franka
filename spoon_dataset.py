import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def data_normalize(data, normalization_params, data_category='poses'):
    """
    Normalize the data using the provided min and max values to scale to [-1, 1].

    Args:
        data (np.ndarray): Data to normalize.
        min_vals (np.ndarray): Min values for normalization.
        max_vals (np.ndarray): Max values for normalization.

    Returns:
        np.ndarray: Normalized data.
    """
    if data_category =='poses':
        min_vals = normalization_params['spoon_poses_min']
        max_vals = normalization_params['spoon_poses_max']
    elif data_category =='joints':
        min_vals = normalization_params['robot_joints_min']
        max_vals = normalization_params['robot_joints_max']
    else:
        print('not valid data_category.')

    # print('normalize data')
    return 2 * (data - min_vals) / (max_vals - min_vals) - 1


def data_inv_normalize(data, normalization_params, data_category='poses'):

    if data_category =='poses':
        min_vals = normalization_params['spoon_poses_min']
        max_vals = normalization_params['spoon_poses_max']
    elif data_category =='joints':
        min_vals = normalization_params['robot_joints_min']
        max_vals = normalization_params['robot_joints_max']
    else:
        print('not valid data_category.')

    print('inverese normalize data')
    return (data + 1) * (max_vals - min_vals) / 2 + min_vals


class ProcessedDataset(Dataset):
    def __init__(self, dataset_path, subsequence_length=16, normalization=True,img_data=False,raw_img=False):
        """
        Initializes the dataset by loading the processed data and preparing it for use.

        Args:
            dataset_path (str): Path to the processed HDF5 file.
            subsequence_length (int): Length of each subsequence.
            normalization (bool): Whether to normalize the spoon poses and robot joints.
        """
        self.h5_file_path = dataset_path
        self.subsequence_length = subsequence_length
        self.normalization = normalization
        self.img_data = img_data
        self.raw_img = raw_img
        self.crop_img_mask = [278, 458, 310, 550]

        self.device = "cuda"
        self.data = self.load_data()
        self.normalization_params = self.calculate_normalization_params() if normalization else None
        self.subsequences = self.create_subsequences()




    def load_data(self):
        """
        Load the processed data from the HDF5 file.

        Returns:
            dict: A dictionary containing the loaded spoon poses, masked images, and robot joints.
        """
        data = {}
        with h5py.File(self.h5_file_path, 'r') as f:
            for episode in f.keys():
                spoon_poses = f[episode]['spoon_poses'][:]
                masked_images = f[episode]['masked_images'][:] / 255
                robot_joints = f[episode]['robot_joints'][:]

                # Convert structured array to a regular ndarray
                spoon_poses = np.stack([pose for pose in spoon_poses['pose']])[:, :, 0]

                # import cv2
                # cv2.imshow('Image', cropped_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if self.img_data:
                    if self.raw_img:

                        data[episode] = {
                            'spoon_poses': spoon_poses,
                            'masked_images': masked_images,
                            'robot_joints': robot_joints
                        }


                    else:

                        # Apply the fixed mask to crop all images
                        cropped_images = []
                        for img in masked_images:
                            cropped_image = img[self.crop_img_mask[0]:self.crop_img_mask[1],
                                            self.crop_img_mask[2]:self.crop_img_mask[3]]
                            cropped_images.append(cropped_image)
                        cropped_images = np.array(cropped_images)

                        data[episode] = {
                            'spoon_poses': spoon_poses,
                            'masked_images': cropped_images,
                            'robot_joints': robot_joints
                        }

                else:
                    data[episode] = {
                        'spoon_poses': spoon_poses,
                        'robot_joints': robot_joints
                    }
            return data

    def calculate_normalization_params(self):
        """
        Calculate min and max for each dimension in spoon poses and robot joints for normalization.

        Returns:
            dict: A dictionary containing the min and max for spoon poses and robot joints.
        """
        all_spoon_poses = []
        all_robot_joints = []

        for episode_data in self.data.values():
            all_spoon_poses.append(episode_data['spoon_poses'])
            all_robot_joints.append(episode_data['robot_joints'])

        all_spoon_poses = np.concatenate(all_spoon_poses, axis=0)
        all_robot_joints = np.concatenate(all_robot_joints, axis=0)

        spoon_poses_min = np.min(all_spoon_poses, axis=0)
        spoon_poses_max = np.max(all_spoon_poses, axis=0)
        robot_joints_min = np.min(all_robot_joints, axis=0)
        robot_joints_max = np.max(all_robot_joints, axis=0)

        # Calculate the extended range
        spoon_poses_range = spoon_poses_max - spoon_poses_min
        robot_joints_range = robot_joints_max - robot_joints_min

        spoon_poses_min_ext = spoon_poses_min - 0.1 * spoon_poses_range
        spoon_poses_max_ext = spoon_poses_max + 0.1 * spoon_poses_range
        robot_joints_min_ext = robot_joints_min - 0.1 * robot_joints_range
        robot_joints_max_ext = robot_joints_max + 0.1 * robot_joints_range

        normalization_params = {
            'spoon_poses_min': spoon_poses_min_ext,
            'spoon_poses_max': spoon_poses_max_ext,
            'robot_joints_min': robot_joints_min_ext,
            'robot_joints_max': robot_joints_max_ext,
        }

        return normalization_params


    def create_subsequences(self):
        """
        Create subsequences of fixed length from the episode data.

        Returns:
            list: A list of subsequences, each containing spoon poses, masked images, robot joints, and masks.
        """
        subsequences_interval = 4
        subsequences = []
        for episode, episode_data in self.data.items():
            # (Your subsequence creation code remains the same, with the addition of masks)
            # Clip off the first 2 steps as CV module need initialize
            spoon_poses = episode_data['spoon_poses'][2:]
            robot_joints = episode_data['robot_joints'][2:]

            if self.normalization:
                spoon_poses = data_normalize(spoon_poses,self.normalization_params,'poses')
                robot_joints = data_normalize(robot_joints,self.normalization_params,'joints')


            num_subsequences = (len(spoon_poses) - self.subsequence_length) // subsequences_interval + 1

            if self.img_data:

                masked_images = episode_data['masked_images'][2:]
                for i in range(0, num_subsequences * subsequences_interval, subsequences_interval):
                    subsequence = {
                        'spoon_poses': torch.tensor(spoon_poses[i:i + self.subsequence_length], dtype=torch.float32),
                        'masked_images': torch.tensor(masked_images[i:i + self.subsequence_length], dtype=torch.float32),
                        'robot_joints': torch.tensor(robot_joints[i:i + self.subsequence_length], dtype=torch.float32),
                    }
                    subsequences.append(subsequence)
            else:

                for i in range(0, num_subsequences * subsequences_interval, subsequences_interval):
                    subsequence = {
                        'spoon_poses': torch.tensor(spoon_poses[i:i + self.subsequence_length], dtype=torch.float32),
                        'robot_joints': torch.tensor(robot_joints[i:i + self.subsequence_length], dtype=torch.float32),
                    }
                    subsequences.append(subsequence)

        print(f'create dataset with image:{self.img_data}, normalization:{self.normalization},subseq interval:{subsequences_interval}, total num of seq:{len(subsequences)}')

        return subsequences

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, index):
        subsequence = self.subsequences[index]

        return subsequence




class MergedDataset(Dataset):
    def __init__(self, dataset_path_A, dataset_path_B, subsequence_length=16, normalization=True, img_data=False,raw_img=False):
        """
        Initializes the dataset by loading the processed data from two sources and preparing it for use.

        Args:
            dataset_path_A (str): Path to the processed HDF5 file with full information.
            dataset_path_B (str): Path to the processed HDF5 file with dummy robot joints.
            subsequence_length (int): Length of each subsequence.
            normalization (bool): Whether to normalize the spoon poses and robot joints.
        """
        self.dataset_path_A = dataset_path_A
        self.dataset_path_B = dataset_path_B
        self.subsequence_length = subsequence_length
        self.normalization = normalization
        self.img_data = img_data
        self.raw_img = raw_img
        self.device = "cuda"

        self.crop_img_mask = [278, 458, 310, 550]

        # Load data from both datasets
        self.data_A = self.load_data(self.dataset_path_A)
        self.data_B = self.load_data(self.dataset_path_B)

        # Create masks
        self.mask_A = 0
        self.mask_B = 1

        # Merge datasets
        self.data, self.masks = self.merge_datasets()

        # Calculate normalization parameters
        self.normalization_params = self.calculate_normalization_params() if normalization else None

        # Create subsequences
        self.subsequences = self.create_subsequences()


    def load_data(self, dataset_path):
        """
        Load the processed data from the HDF5 file.

        Returns:
            dict: A dictionary containing the loaded spoon poses, masked images, and robot joints.
        """
        data = {}
        with h5py.File(dataset_path, 'r') as f:
            for episode in f.keys():

                spoon_poses = f[episode]['spoon_poses'][:]
                masked_images = f[episode]['masked_images'][:]/255
                robot_joints = f[episode]['robot_joints'][:]

                # Convert structured array to a regular ndarray
                spoon_poses = np.stack([pose for pose in spoon_poses['pose']])[:, :, 0]

                # import cv2
                # cv2.imshow('Image', cropped_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if self.img_data:
                    if self.raw_img:

                        data[episode] = {
                            'spoon_poses': spoon_poses,
                            'masked_images': masked_images,
                            'robot_joints': robot_joints
                        }


                    else:

                        # Apply the fixed mask to crop all images
                        cropped_images = []
                        for img in masked_images:
                            cropped_image = img[self.crop_img_mask[0]:self.crop_img_mask[1],
                                            self.crop_img_mask[2]:self.crop_img_mask[3]]
                            cropped_images.append(cropped_image)
                        cropped_images = np.array(cropped_images)

                        data[episode] = {
                            'spoon_poses': spoon_poses,
                            'masked_images': cropped_images,
                            'robot_joints': robot_joints
                        }

                else:
                    data[episode] = {
                        'spoon_poses': spoon_poses,
                        'robot_joints': robot_joints
                    }
        return data

    def merge_datasets(self):
        """
        Merge the data from the two datasets, adding masks.

        Returns:
            tuple: A tuple containing the merged data and corresponding masks.
        """
        merged_data = {}
        masks = {}

        # Add data from A with mask 0
        for episode, data in self.data_A.items():
            merged_data[episode] = data
            masks[episode] = self.mask_A

        # Add data from B with mask 1
        for episode, data in self.data_B.items():
            merged_data[f"B_{episode}"] = data  # Prefix B to avoid key conflicts
            masks[f"B_{episode}"] = self.mask_B

        return merged_data, masks

    def calculate_normalization_params(self):
        """
        Calculate min and max for each dimension in spoon poses and robot joints for normalization.

        Returns:
            dict: A dictionary containing the min and max for spoon poses and robot joints.
        """
        all_spoon_poses = []
        all_robot_joints = []

        for episode_data in self.data.values():
            all_spoon_poses.append(episode_data['spoon_poses'])
            all_robot_joints.append(episode_data['robot_joints'])

        all_spoon_poses = np.concatenate(all_spoon_poses, axis=0)
        all_robot_joints = np.concatenate(all_robot_joints, axis=0)

        spoon_poses_min = np.min(all_spoon_poses, axis=0)
        spoon_poses_max = np.max(all_spoon_poses, axis=0)
        robot_joints_min = np.min(all_robot_joints, axis=0)
        robot_joints_max = np.max(all_robot_joints, axis=0)

        # Calculate the extended range
        spoon_poses_range = spoon_poses_max - spoon_poses_min
        robot_joints_range = robot_joints_max - robot_joints_min

        spoon_poses_min_ext = spoon_poses_min - 0.1 * spoon_poses_range
        spoon_poses_max_ext = spoon_poses_max + 0.1 * spoon_poses_range
        robot_joints_min_ext = robot_joints_min - 0.1 * robot_joints_range
        robot_joints_max_ext = robot_joints_max + 0.1 * robot_joints_range

        normalization_params = {
            'spoon_poses_min': spoon_poses_min_ext,
            'spoon_poses_max': spoon_poses_max_ext,
            'robot_joints_min': robot_joints_min_ext,
            'robot_joints_max': robot_joints_max_ext,
        }

        return normalization_params

    def create_subsequences(self):
        """
        Create subsequences of fixed length from the episode data.

        Returns:
            list: A list of subsequences, each containing spoon poses, masked images, robot joints, and masks.
        """
        subsequences_interval = 2
        subsequences = []
        for episode, episode_data in self.data.items():
            mask = self.masks[episode]
            # (Your subsequence creation code remains the same, with the addition of masks)
            # Clip off the first 2 steps as CV module need initialize
            spoon_poses = episode_data['spoon_poses'][2:]
            robot_joints = episode_data['robot_joints'][2:]

            if self.normalization:
                spoon_poses = data_normalize(spoon_poses,self.normalization_params,'poses')
                robot_joints = data_normalize(robot_joints,self.normalization_params,'joints')


            num_subsequences = (len(spoon_poses) - self.subsequence_length) // subsequences_interval + 1

            if self.img_data:

                masked_images = episode_data['masked_images'][2:]
                for i in range(0, num_subsequences * subsequences_interval, subsequences_interval):
                    subsequence = {
                        'spoon_poses': torch.tensor(spoon_poses[i:i + self.subsequence_length], dtype=torch.float32),
                        'masked_images': torch.tensor(masked_images[i:i + self.subsequence_length], dtype=torch.float32),
                        'robot_joints': torch.tensor(robot_joints[i:i + self.subsequence_length], dtype=torch.float32),
                        'mask': torch.tensor(mask, dtype=torch.float32)
                    }
                    subsequences.append(subsequence)
            else:

                for i in range(0, num_subsequences * subsequences_interval, subsequences_interval):
                    subsequence = {
                        'spoon_poses': torch.tensor(spoon_poses[i:i + self.subsequence_length], dtype=torch.float32),
                        'robot_joints': torch.tensor(robot_joints[i:i + self.subsequence_length], dtype=torch.float32),
                        'mask': torch.tensor(mask, dtype=torch.float32)
                    }
                    subsequences.append(subsequence)

        print(f'create dataset with image:{self.img_data}, normalization:{self.normalization},subseq interval:{subsequences_interval}, total num of seq:{len(subsequences)}')

        return subsequences

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, index):
        subsequence = self.subsequences[index]

        return subsequence

# Example usage
if __name__ == "__main__":
    dataset_path_A = 'wood_spoon/train_robot_dataset.h5'
    dataset_path_B = 'wood_spoon/train_human_dataset.h5'
    dataset = MergedDataset(dataset_path_A, dataset_path_B, subsequence_length=16, normalization=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=1)

    for batch in dataloader:
        spoon_poses = batch['spoon_poses'].cuda()
        robot_joints = batch['robot_joints'].cuda()
        mask = batch['mask'].cuda()

        print(spoon_poses[0].shape)

        # Your training logic here
        # model_output = model(spoon_poses, robot_joints, mask)



# Example usage
# if __name__ == "__main__":
#     h5_file_path = 'wood_spoon/processed_robot_data.h5'  # Adjust the file path as necessary
#     dataset = ProcessedDataset(h5_file_path, subsequence_length=16, normalization=True)
#     print('dataset created, loading data')
#     # Use DataLoader to handle batching and moving data to the GPU during training
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True,pin_memory=True,num_workers=1)
#
#     # Example training loop
#     for batch in dataloader:
#         # Move batch to GPU
#         spoon_poses = batch['spoon_poses'].cuda()
#         masked_images = batch['masked_images'].cuda()
#         robot_joints = batch['robot_joints'].cuda()
#
#         print(spoon_poses[0].shape)
#
#         # Your training logic here
#         # model_output = model(spoon_poses, masked_images, robot_joints)
