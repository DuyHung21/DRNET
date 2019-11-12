import os

import cv2
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MovingMnist(Dataset):
    
    def __init__(self, root_dir=None, root_file=None, is_file=True, transform=None):
        """
        Args:
            root_dir (string): Directory to all video, if data are videos
            root_file (string): Directory to the numpy file that store the data
            is_file (boolean): Wheter data are read from file or from folder
        """

        self.data = None
        self.transform = transform
        if is_file:
            self.data = np.load(root_file)
    
    def __len__(self):
        return self.data.shape[1]


    def __getitem__(self, index):
        """
            Args: 
                index (number): index of the video
            Return:
                xi_t: frame t in video i
                xi_tk: frame t+k in video i
                xj_tk: frame t+k in video j
        """

        num_seq = self.data.shape[1]
        num_frame = self.data.shape[0]

        t = np.random.randint(num_frame)
        while t == num_frame - 1:
            t = np.random.randint(num_frame)

        k = np.random.randint(num_frame - t - 1) + 1

        j = np.random.randint(num_seq)
        while j == index:
            j = np.random.randint(num_seq)
        
        out = [self.data[t, index], self.data[t+k, index], self.data[t+k, j]]
        if self.transform is not None:
            out = [self.transform(x) for x in out]

        return out

if __name__ == '__main__':
    dataset = MovingMnist(root_file='../../dataset/mnist_test_seq.npy',
                            transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                            ])
                        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=True, num_workers=4)

    print(len(dataloader))
    real_batch = next(iter(dataloader))
    fig = plt.figure(figsize=(12,12))
    plt.axis("off")
    plt.title("Training Images")

    plt.subplot(2, 2, 1)
    xi_t = vutils.make_grid(real_batch[0][:64], padding=2)
    plt.imshow(np.transpose(xi_t,(1,2,0)))

    plt.subplot(2, 2, 2)
    xi_tk = vutils.make_grid(real_batch[1][:64], padding=2)
    plt.imshow(np.transpose(xi_tk,(1,2,0)))

    plt.subplot(2, 2, 3)
    xj_tk = vutils.make_grid(real_batch[2][:64], padding=2)
    plt.imshow(np.transpose(xj_tk,(1,2,0)))

    plt.show()