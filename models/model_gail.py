"""
Copyright Notice:
    The implementation of our GAIL network was created based on the following public repository:
    https://github.com/Khrylx/PyTorch-RL
"""

import math
import os
import glob
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

FEAT_DIM = 512
IMAGE_DIM = 3
ACTION_DIM = 2


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


#............................................... Policy network / Generator ...............................................


class Policy(nn.Module):

    def __init__(self, image_dim=IMAGE_DIM, action_dim=ACTION_DIM, feat_dim=FEAT_DIM, log_std=0):
        
        super().__init__()

        # Creating feature extractor model
        resnet = models.resnet50(pretrained=True).eval()
        modules = list(resnet.children())[:-4]          # Remove the last 2 layers (FC and AveragePooling) and the last ResNet 2 blocks
        self.feature_extractor = nn.Sequential(*modules)
        for p in self.feature_extractor.parameters():   # Set the feature extractor network's weights to non-trainable, so that the
            p.requires_grad = False                     # gradients of the feature tensors will not cause an error

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(1536, 256),
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer2 = nn.Linear(256, 128)

        self.action_mean = nn.Linear(128, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)


    def forward(self, input_img):
        
        input_feat = self.feature_extractor(input_img)
        out = self.conv_layer1(input_feat)      # 1st Convolutional Layer (stride of 1)
        out = self.conv_layer2(out)             # 2nd Convolutional Layer (stried of 1)
        out = out.reshape(out.size(0), -1)      # Substitutes Flatten Layer
        out = self.fc_layer1(out)               # 1st Dense Layer of the image features (256)
        out = self.fc_layer2(out)               # 2nd Dense Layer of the image features (128)
        out = F.leaky_relu(out, 0.3)            # LeakyReLU
        action_mean = self.action_mean(out)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std


    def predict(self, input_img):

        input_img = torch.Tensor(input_img).float().unsqueeze(0)    # Torch Tensor: [1, 40, 80, 3] or [1, 40, 80, 1]
        input_img = input_img.permute(0, 3, 1, 2)                   # Torch Tensor: [1, 3, 40, 80] or [1, 1, 40, 80] 
        action, _, _ = self.forward(input_img)
        
        return action.squeeze().detach().cpu().numpy()


    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action


    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)


    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


#....................................................... Discriminator .......................................................


class Discriminator(nn.Module):

    def __init__(self, image_dim=IMAGE_DIM, action_dim=ACTION_DIM):

        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(image_dim, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0), 
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(4608 + action_dim, 256),
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.3))

        self.logic = nn.Linear(128, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)


    def forward(self, input_img, input_action):

        out = self.conv_layer1(input_img)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out_img = out.reshape(out.size(0), -1)
        out = torch.cat((out_img, input_action), dim=1)
        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        prob = torch.sigmoid(self.logic(out))
        return prob


#...................................................... Value network ......................................................


class Value(nn.Module):

    def __init__(self, image_dim=IMAGE_DIM):

        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(image_dim, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0), 
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(4608 , 256),
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.3))

        self.value_head = nn.Linear(128, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)


    def forward(self, input_img):

        out = self.conv_layer1(input_img)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        value = self.value_head(out)
        return value


#........................................ Behavioural Cloning pretrainer for the policy ........................................


class PolicyPretrainer:
    def __init__(self, learning_rate=3e-4):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FeatureBasedPolicy(log_std=-0.0).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        model_save_path = 'trained_models/gail' 
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            print('\nFolder created: ', model_save_path)

    def train(self, feature_batch, action_batch):

        # Permutation not needed, it is already done before the image is fed trough the feature extractor, the output of it
        # is therefore already in channel first format
        feature_batch = torch.Tensor(feature_batch).float().to(self.device)  
        action_batch = torch.Tensor(action_batch).float().to(self.device)

        self.optimizer.zero_grad()
        output, _, _ = self.model(feature_batch)
        loss = self.criterion(output.squeeze(), action_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_validation_loss(self, feature_batch, action_batch):
        
        # Permutation not needed, it is already done before the image is fed trough the feature extractor, the output of it
        # is therefore already in channel first format
        feature_batch = torch.Tensor(feature_batch).float().to(self.device) 
        action_batch = torch.Tensor(action_batch).float().to(self.device)

        with torch.no_grad():
            
            output, _, _ = self.model(feature_batch)
            loss = self.criterion(output.squeeze(), action_batch)

        return loss.item()

    def save(self):
        torch.save(self.model.state_dict(), 'trained_models/gail/policy_net.pth')

    def save_epoch(self, epoch_number):
        path = 'trained_models/gail/policy_net_epoch_{}.pth'.format(epoch_number)
        torch.save(self.model.state_dict(), path)

    def rename_best_epoch(self, epoch_number):
        old_name = 'trained_models/gail/policy_net.pth'
        new_name = 'trained_models/gail/policy_net_best_epoch_{}.pth'.format(epoch_number)
        os.rename(old_name, new_name)
    
    def load(self):
        model_path = "trained_models/gail/policy_net_best_epoch_*.pth"
        best_model_path = glob.glob(model_path)[0]
        self.model.to(torch.device('cpu'))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        return self.model


#................................... Policy network that expects image features as an input ...................................


class FeatureBasedPolicy(nn.Module):

    def __init__(self, action_dim=ACTION_DIM, feat_dim=FEAT_DIM, log_std=0):
        
        super().__init__()

        # Creating feature extractor model
        resnet = models.resnet50(pretrained=True).eval()
        modules = list(resnet.children())[:-4]          # Remove the last 2 layers (FC and AveragePooling) and the last ResNet 2 blocks
        self.feature_extractor = nn.Sequential(*modules)
        for p in self.feature_extractor.parameters():   # Set the feature extractor network's weights to non-trainable, so that the
            p.requires_grad = False                     # gradients of the feature tensors will not cause an error

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer1 = nn.Sequential(
            nn.Linear(1536, 256),
            nn.LeakyReLU(negative_slope=0.3))

        self.fc_layer2 = nn.Linear(256, 128)

        self.action_mean = nn.Linear(128, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)


    def forward(self, input_feat):
        
        out = self.conv_layer1(input_feat)      # 1st Convolutional Layer (stride of 1)
        out = self.conv_layer2(out)             # 2nd Convolutional Layer (stried of 1)
        out = out.reshape(out.size(0), -1)      # Substitutes Flatten Layer
        out = self.fc_layer1(out)               # 1st Dense Layer of the image features (256)
        out = self.fc_layer2(out)               # 2nd Dense Layer of the image features (128)
        out = F.leaky_relu(out, 0.3)            # LeakyReLU
        action_mean = self.action_mean(out)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std
