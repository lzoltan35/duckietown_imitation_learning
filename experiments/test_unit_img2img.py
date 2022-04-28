"""
Copyright Notice:
    The implementation of our UNIT network was created based on the following public repository:
    https://github.com/eriklindernoren/PyTorch-GAN#unit
"""

import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.model_unit_network import *
from utils.unit_utils import *


# Sets how many image-to-image translation samples do we want to generate
NUMBER_OF_SAMPLES = 20


epoch = 199                 # epoch to visualize
batch_size = 1              # batch size
lr = 0.0001                 # adam learning rate
b1 = 0.5                    # adam: decay of first order momentum of gradient
b2 = 0.999                  # adam: decay of first order momentum of gradient
decay_epoch = 100           # epoch from which to start lr decay
n_cpu = 8                   # number of cpu threads to use during batch generation
img_height = 60             # size of image height
img_width = 80              # size of image width
img_channels = 3            # number of image channels
sample_interval = 1024      # interval between saving generator samples
checkpoint_interval = 1     # interval between saving model checkpoints
n_downsample = 2            # number downsampling layers in encoder
n_dim = 64                  # number of filters in first encoder layer

cuda = True if torch.cuda.is_available() else False

model_save_path = 'trained_models/unit'

# Create sample directory
os.makedirs("unit_img2img/test", exist_ok=True)

input_shape = (img_channels, img_height, img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = n_dim * 2 ** n_downsample

# Initialize generator and discriminator
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(dim=n_dim, n_downsample=n_downsample, shared_block=shared_E)
E2 = Encoder(dim=n_dim, n_downsample=n_downsample, shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(dim=n_dim, n_upsample=n_downsample, shared_block=shared_G)
G2 = Generator(dim=n_dim, n_upsample=n_downsample, shared_block=shared_G)
D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)

if cuda:
    E1 = E1.cuda()
    E2 = E2.cuda()
    G1 = G1.cuda()
    G2 = G2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()

if epoch != 0:
    # Load pretrained models
    E1.load_state_dict(torch.load("%s/E1_%d.pth" % (model_save_path, epoch)))
    E2.load_state_dict(torch.load("%s/E2_%d.pth" % (model_save_path, epoch)))
    G1.load_state_dict(torch.load("%s/G1_%d.pth" % (model_save_path, epoch)))
    G2.load_state_dict(torch.load("%s/G2_%d.pth" % (model_save_path, epoch)))
    D1.load_state_dict(torch.load("%s/D1_%d.pth" % (model_save_path, epoch)))
    D2.load_state_dict(torch.load("%s/D2_%d.pth" % (model_save_path, epoch)))
else:
    # Initialize weights
    E1.apply(weights_init_normal)
    E2.apply(weights_init_normal)
    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Test data loader
val_dataloader = DataLoader(
    ImageDataset("./unit_data/sim2real", transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    X1 = Variable(imgs["A"].type(Tensor))
    X2 = Variable(imgs["B"].type(Tensor))
    _, Z1 = E1(X1)
    _, Z2 = E2(X2)
    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)
    img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
    save_image(img_sample, "unit_img2img/test/%s.png" % batches_done, nrow=5, normalize=True)

for i in range(NUMBER_OF_SAMPLES):
    sample_images('test_{}'.format(i))
    print(f'{i}/{NUMBER_OF_SAMPLES}')
