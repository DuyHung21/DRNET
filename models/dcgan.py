from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, d_in, d_out):
        """
        Encoder of Dcgan


        Args:
            d_in: Input dimension
        """

        # Encoder has 5 CNN layers.
        # Each CNN layer (execpt the last) is followed by a BN and a Leaky Relu layer
        # After the last CNN layer, normalize the vector have unit norm

        super(Encoder, self).__init__()

        encoder_layers = []

        encode_d_in = d_in
        encode_d_out = 128

        for i in range(4):
            encoder_layers.append(
                ('conv_{}'.format(i+1), nn.Conv2d(encode_d_in, encode_d_out, 4, 2, 1))
            )
            encoder_layers.append(
                ('bn_{}'.format(i+1), nn.BatchNorm2d(encode_d_out))
            )
            encoder_layers.append(
                ('act_{}'.format(i+1), nn.LeakyReLU())
            )

            encode_d_in = encode_d_out
            encode_d_out = 2 * encode_d_out

        encoder_layers.append(
            ('conv_{}'.format(5), nn.Conv2d(encode_d_in, d_out, 4, 2, 1))
        )
        encoder_layers.append(
            ('act_{}'.format(5), nn.Tanh())
        )

        self.encoder_layers = nn.Sequential(OrderedDict(encoder_layers))
    
    def forward(self, x):
        out = self.encoder_layers(x)
        out = F.normalize(out)

        return out


class Decoder(nn.Module):
    def __init__(self, d_in, d_out):
        """
        Decoder of Dcgan


        Args:
            d_in: Input dimension
        """

        # Decoder is the mirror version of the Encoder

        super(Decoder, self).__init__()

        decode_layers = []

        decode_d_in = d_in
        decode_d_out = 1024


        for i in range(4):
            decode_layers.append(
                ('conv_{}'.format(i+1), nn.ConvTranspose2d(decode_d_in, decode_d_out, 4, 2, 1))
            )
            decode_layers.append(
                ('bn_{}'.format(i+1), nn.BatchNorm2d(decode_d_out))
            )
            decode_layers.append(
                ('act_{}'.format(i+1), nn.LeakyReLU())
            )

            decode_d_in = decode_d_out
            decode_d_out = decode_d_out // 2

        decode_layers.append(
            ('conv_{}'.format(5), nn.ConvTranspose2d(decode_d_in, d_out, 4, 2, 1))
        )
        decode_layers.append(
            ('act_{}'.format(5), nn.Sigmoid())
        )

        self.decode_layers = nn.Sequential(OrderedDict(decode_layers))
    
    def forward(self, content, pose):
        inp1 = torch.cat([content, pose], dim=1)
        out = self.decode_layers(inp1)

        return out


# testing code
if __name__ == '__main__':
    inp = torch.randn(4,3,256,256)
    
    net = Encoder(3,128)
    out = net(inp)
    print('content shape', out.shape)

    net = Encoder(3, 10)
    pose = net(inp)
    print('pose shape', pose.shape)

    net = Decoder(128+10, 3)
    out = net(out, pose)
    print('decode shape', out.shape)

    
        
        