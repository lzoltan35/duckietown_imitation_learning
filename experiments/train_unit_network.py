"""
Copyright Notice:
    The implementation of our UNIT network was created based on the following public repository:
    https://github.com/eriklindernoren/PyTorch-GAN#unit
"""

import os
import numpy as np
import itertools
import datetime
import time
import sys

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.model_unit_network import *
from utils.unit_utils import *



epoch = 0                   # epoch to start training from
n_epochs = 200              # number of epochs of training
batch_size = 4              # batch size
lr = 0.0001                 # adam learning rate
b1 = 0.5                    # adam: decay of first order momentum of gradient
b2 = 0.999                  # adam: decay of first order momentum of gradient
decay_epoch = 100           # epoch from which to start lr decay
n_cpu = 8                   # number of cpu threads to use during batch generation
img_height = 60             # size of image height
img_width = 80              # size of image width
img_channels = 3            # number of image channels
sample_interval = 256       # interval between saving generator samples
checkpoint_interval = 1     # interval between saving model checkpoints
n_downsample = 2            # number downsampling layers in encoder
n_dim = 64                  # number of filters in first encoder layer

cuda = True if torch.cuda.is_available() else False

model_save_path = 'trained_models/unit'

# Create sample and checkpoint directories
os.makedirs("unit_img2img", exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixel = torch.nn.L1Loss()

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
    criterion_GAN.cuda()
    criterion_pixel.cuda()

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

# Loss weights
lambda_0 = 10  # GAN
lambda_1 = 0.1  # KL (encoded images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated images)
lambda_4 = 100  # Cycle pixel-wise

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
    lr=lr,
    betas=(b1, b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=lr, betas=(b1, b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=lr, betas=(b1, b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("./unit_data/sim2real", transforms_=transforms_, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)
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
    save_image(img_sample, "unit_img2img/%s.png" % int(batches_done/sample_interval), nrow=5, normalize=True)


def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((X1.size(0), *D1.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((X1.size(0), *D1.output_shape))), requires_grad=False)

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        mu1, Z1 = E1(X1)
        mu2, Z2 = E2(X2)

        # Reconstruct images
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)

        # Translate images
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        # Cycle translation
        mu1_, Z1_ = E1(fake_X1)
        mu2_, Z2_ = E2(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)

        # Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_pixel(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_pixel(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_pixel(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_pixel(cycle_X2, X2)

        # Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(E1.state_dict(), "%s/E1_%d.pth" % (model_save_path, epoch))
        torch.save(E2.state_dict(), "%s/E2_%d.pth" % (model_save_path, epoch))
        torch.save(G1.state_dict(), "%s/G1_%d.pth" % (model_save_path, epoch))
        torch.save(G2.state_dict(), "%s/G2_%d.pth" % (model_save_path, epoch))
        torch.save(D1.state_dict(), "%s/D1_%d.pth" % (model_save_path, epoch))
        torch.save(D2.state_dict(), "%s/D2_%d.pth" % (model_save_path, epoch))
