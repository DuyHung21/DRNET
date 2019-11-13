import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.utils as vutils
import torchvision.transforms as transforms

from data.mnist import MovingMnist
from models.scene_discriminator import SceneDiscriminator
from models import dcgan as DcGan
from models.lstm import LSTM

import utils as nutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_value(tensor):
    if tensor is not None:
        return tensor.item()
    else:
        return 0


def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False


def unfreeze_network(network):
    for param in network.parameters():
        param.requires_grad = True


def train_scene_discriminator(xi_t, xi_tk, xj_tk):
    """
    Args:
        xi_t: frame t from video i
        xi_tk: frame t + k from video i
        xj_tk: frame t + k from video j
    """
    discriminator.zero_grad()

    # Compute pose vectors
    pi_t = pose_encoder(xi_t)
    pi_tk = pose_encoder(xi_tk)
    pj_tk = pose_encoder(xj_tk)

    # Compute the output of discriminator C
    pred_same = discriminator(pi_t, pi_tk)
    pred_diff = discriminator(pi_t, pj_tk).detach()

    loss, acc = nutils.discriminator_loss(pred_same, pred_diff, device=device)
    loss.backward()
    discriminator_optim.step()

    return get_value(loss), get_value(acc)


def train_main_network(xi_t, xi_tk, xj_tk):
    content_encoder.zero_grad()
    pose_encoder.zero_grad()
    decoder.zero_grad()

    # Compute content vectors of video i
    ci_t = content_encoder(xi_t)
    ci_tk = content_encoder(xi_tk).detach()

    # Compute pose vectors of video i
    pi_t = pose_encoder(xi_t)
    pi_tk = pose_encoder(xi_tk).detach()

    # Compute pose vector of video j 
    pj_tk = pose_encoder(xj_tk).detach()

    # Compuse scene discrimination vector
    discr_same = discriminator(pi_t, pi_tk)
    discr_diff = discriminator(pi_t, pj_tk)

    # Compute reconsctruct image
    pred_xitk = decoder(ci_t, pi_tk)

    # Similarity loss
    sim_loss = nutils.similarity_loss(ci_t, ci_tk, device=device)

    # Reconstruction loss
    rec_loss = nutils.reconstruction_loss(pred_xitk, xi_tk, device=device)

    # Adversarial loss
    adv_loss = nutils.adversarial_loss(discr_same, discr_diff, device=device)

    # Total loss
    loss = rec_loss + alpha * sim_loss + beta * adv_loss

    loss.backward()

    pose_encoder_optim.step()
    content_encoder_optim.step()
    decoder_optim.step()

    return get_value(rec_loss), get_value(sim_loss), get_value(adv_loss)



if __name__ == "__main__":
    batch_size = 32
    lr = 0.002
    alpha = 1
    beta = 0.1
    max_iter = 200
    save_iters = 5
    start = 0

    # Define networks
    content_encoder = DcGan.Encoder(d_in=1, d_out=128)
    content_encoder_optim = optim.Adam(content_encoder.parameters(), lr=lr)

    pose_encoder = DcGan.Encoder(d_in=1, d_out=5)
    pose_encoder_optim = optim.Adam(pose_encoder.parameters(), lr=lr)

    decoder = DcGan.Decoder(d_in=128+5, d_out=1)
    decoder_optim = optim.Adam(decoder.parameters(), lr=lr)

    discriminator = SceneDiscriminator(d_in=5 * 2 * 2)
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr)

    # content_encoder.load_state_dict(torch.load('pretrained/iter_50/content_encoder.pth', map_location=device))
    # pose_encoder.load_state_dict(torch.load('pretrained/iter_50/pose_encoder.pth', map_location=device))
    # decoder.load_state_dict(torch.load('pretrained/iter_50/decoder.pth', map_location=device))
    # discriminator.load_state_dict(torch.load('pretrained/iter_50/scene_discriminator.pth', map_location=device))
    # start = 50

    lstm = LSTM(128+5, 256, 5, batch_size, 2)

    # Use GPU if available
    content_encoder = content_encoder.to(device)
    pose_encoder =  pose_encoder.to(device)
    decoder = decoder.to(device)
    discriminator = discriminator.to(device)
    lstm = lstm.to(device)

    dataset = MovingMnist(root_file='../dataset/mnist_test_seq.npy',
                            transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                            ])
                        )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

    for i in range(start, max_iter):
        content_encoder.train()
        pose_encoder.train()
        decoder.train()
        discriminator.train()

        avg_dis, avg_sim, avg_rec, avg_adv = 0, 0, 0, 0

        num_data = len(dataloader)
        for di, (xi_t, xi_tk, xj_tk) in enumerate(dataloader):
            # Use GPU if needed
            xi_t = xi_t.to(device)
            xi_tk = xi_tk.to(device)
            xj_tk = xj_tk.to(device)

            # train discriminator
            dis_loss, acc = train_scene_discriminator(xi_t, xi_tk, xj_tk)

            # train main network
            rec_loss, sim_loss, adv_loss= train_main_network(xi_t, xi_tk, xj_tk)

            # display losses
            
            if di % 50 == 0:
                print("%d/%d - discr: %0.3f, rec: %0.3f, sim: %0.3f, adv: %0.3f"%(di, num_data, dis_loss, rec_loss, sim_loss, adv_loss))
        
            avg_dis += dis_loss
            avg_sim += sim_loss
            avg_rec += rec_loss
            avg_adv += adv_loss
        
        print("---------------------------------------")
        print("iteration %d" % i)
        print("discr loss: %0.5f" % (avg_dis / num_data))
        print("rec loss: %0.5f" % (avg_rec / num_data))
        print("sim loss: %0.5f" % (avg_sim / num_data))
        print("adv loss: %0.5f" % (avg_adv / num_data))
        print()

        if i % save_iters == 0:
            save_dir = os.path.join('pretrained', 'iter_%d'%(i)) 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(content_encoder.state_dict(), os.path.join(save_dir,'content_encoder.pth'))
            torch.save(pose_encoder.state_dict(), os.path.join(save_dir,'pose_encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_dir,'decoder.pth'))
            torch.save(discriminator.state_dict(), os.path.join(save_dir,'scene_discriminator.pth'))

