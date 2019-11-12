import os

import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models import dcgan as DcGan
from models.scene_discriminator import SceneDiscriminator
from data.mnist import MovingMnist

content_encoder = DcGan.Encoder(d_in=1, d_out=128)
pose_encoder = DcGan.Encoder(d_in=1, d_out=5)
decoder = DcGan.Decoder(d_in=128+5, d_out=1)
discriminator = SceneDiscriminator(d_in=5 * 2 * 2)

content_encoder.load_state_dict(torch.load('pretrained/iter_50/content_encoder.pth', map_location='cpu'))
pose_encoder.load_state_dict(torch.load('pretrained/iter_50/pose_encoder.pth', map_location='cpu'))
decoder.load_state_dict(torch.load('pretrained/iter_50/decoder.pth', map_location='cpu'))
discriminator.load_state_dict(torch.load('pretrained/iter_50/scene_discriminator.pth', map_location='cpu'))

content_encoder.eval()
pose_encoder.eval()
decoder.eval()
discriminator.eval()


dataset = MovingMnist(root_file='../dataset/mnist_test_seq.npy',
                            transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                            ])
                        )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=True, num_workers=4)


first_batch = next(iter(dataloader))
x = torch.unsqueeze(first_batch[0][0], 0)

hc = content_encoder(x)
hp = pose_encoder(x)
pred_x = decoder(hc, hp)
print(x.shape, torch.squeeze(x, 0).shape)

img = torch.squeeze(x).numpy()
pred_img = torch.squeeze(pred_x).detach().numpy()
print(np.unique(img), np.unique(pred_img))
fig = plt.figure(figsize=(12,12))
plt.axis("off")
plt.title("Testing")
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(pred_img)
plt.show()
