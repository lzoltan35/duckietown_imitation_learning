import torch 
import torch.nn as nn

import numpy as np
import os
import glob

# If this variable is true, the image preprocessing wrapper will not only resize and crop the image, but it will also perform
# a thresholding operation on the image to extract the white and yellow lines
# So if it's True, the agent will learn on a single channel image, if it's false, the agent will learn on images with 3 channels
IMAGE_THRESHOLDING = False

class PytorchTrainer:
    def __init__(self, learning_rate=0.0001):
        
        if IMAGE_THRESHOLDING:
            input_channels = 1
        else:
            input_channels = 3

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PytorchModel(input_img_channels=input_channels, output_size=2).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        model_save_path = 'trained_models/dpl' 
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            print('\nFolder created: ', model_save_path)

    def train(self, observation_batch, action_batch):
        # Input shape: (32, 40, 80, 3) or (32, 40, 80, 1)        
        observation_batch = torch.Tensor(observation_batch).to(self.device).permute(0, 3, 1, 2)    
        # New shape: Torch Tensor: [32, 3, 40, 80] or [32, 1, 40, 80] - PyTorch uses channel first format
        
        action_batch = torch.Tensor(action_batch).to(self.device)

        self.optimizer.zero_grad()
        output = self.model(observation_batch)
        loss = self.criterion(output.squeeze(), action_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_validation_loss(self, observation_batch, action_batch):
        # Input shape: (32, 40, 80, 3) or (32, 40, 80, 1)        
        observation_batch = torch.Tensor(observation_batch).to(self.device).permute(0, 3, 1, 2)    
        # New shape: Torch Tensor: [32, 3, 40, 80] or [32, 1, 40, 80] - PyTorch uses channel first format
        
        action_batch = torch.Tensor(action_batch).to(self.device)

        with torch.no_grad():
            
            output = self.model(observation_batch)
            loss = self.criterion(output.squeeze(), action_batch)

        return loss.item()

    def save(self):
        torch.save(self.model.state_dict(), 'trained_models/dpl/pytorch_convnet.pth')
    
    def save_epoch(self, epoch_number):
        path = 'trained_models/dpl/pytorch_convnet_epoch_{}.pth'.format(epoch_number)
        torch.save(self.model.state_dict(), path)

    def rename_best_epoch(self, epoch_number):
        old_name = 'trained_models/dpl/pytorch_convnet.pth'
        new_name = 'trained_models/dpl/pytorch_convnet_best_epoch_{}.pth'.format(epoch_number)
        os.rename(old_name, new_name)

    def load(self):
        model_path = "trained_models/dpl/pytorch_convnet_best_epoch_*.pth"
        best_model_path = glob.glob(model_path)[0]
        self.model.to(torch.device('cpu'))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        return self.model


class BiggerPytorchModel(nn.Module):
    def __init__(self, input_img_channels=3, output_size=2):
        super(BiggerPytorchModel, self).__init__()

        # Input: 40x80x3 ( H x W x CH )
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_img_channels, 16, kernel_size=5, stride=1, padding=2), # H x W x 16 - this Conv layer keeps the size
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=2, stride=2)) # H/2 x W/2 x 16 - this MaxPool layer reduces the size (divide by 2)
        # Output: 20x40x16 ( H/2 x W/2 x 16 )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # H/2 x W/2 x 32 - this Conv layer keeps the size
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)) # H/2 x W/2 x 32 - this MaxPool layer keeps the size
        # Output: 20x40x32 ( H/2 x W/2 x 32 )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # H/2 x W/2 x 64 - this Conv layer keeps the size
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=2, stride=2)) # H/4 x W/4 x 64 - this MaxPool layer reduces the size (divide by 2)
        # Output: 10x20x64 ( H/4 x W/4 x 64 )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), # H/4 x W/4 x 128 - this Conv layer keeps the size
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)) # H/4 x W/4 x 128 - this MaxPool layer keeps the size
        # Output: 10x20x128 ( H/4 x W/4 x 128 )

        # Here we need to set the parameters according to the Height and Weight sizes of the input picture
        # (Parameters = 128 * H/4 * W/4)
        self.fc = nn.Linear(128 * 10 * 20, output_size)
        
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, input): 
                                                    # Input shape: (40, 80, 3) or (40, 80, 1)
        input = torch.Tensor(input).unsqueeze(0)    # Torch Tensor: [1, 40, 80, 3] or [1, 40, 80, 1]
        input = input.permute(0, 3, 1, 2)           # Torch Tensor: [1, 3, 40, 80] or [1, 1, 40, 80]

        return self.forward(input).squeeze().detach().numpy()


class PytorchModel(nn.Module):
    def __init__(self, input_img_channels=3, output_size=2):
        super(PytorchModel, self).__init__()

        # Input: 40x80x3 ( H x W x CH )
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_img_channels, 16, kernel_size=5, stride=1, padding=2), # H x W x 16 - this Conv layer keeps the size
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=2, stride=2)) # H/2 x W/2 x 16 - this MaxPool layer reduces the size (divide by 2)
        # Output: 20x40x16 ( H/2 x W/2 x 16 )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # H/2 x W/2 x 32 - this Conv layer keeps the size
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=2, stride=2)) # H/4 x W/4 x 32 - this MaxPool layer reduces the size (divide by 2)
        # Output: 10x20x32 ( H/4 x W/4 x 32 )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # H/4 x W/4 x 64 - this Conv layer keeps the size
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)) # H/4 x W/4 x 64 - this MaxPool layer keeps the size
        # Output: 10x20x64 ( H/4 x W/4 x 64 )

        # Here we need to set the parameters according to the Height and Weight sizes of the input picture
        # (Parameters = 64 * H/4 * W/4)
        self.fc = nn.Linear(64 * 10 * 20, output_size)
        
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, input): 
                                                    # Input shape: (40, 80, 3) or (40, 80, 1)
        input = torch.Tensor(input).unsqueeze(0)    # Torch Tensor: [1, 40, 80, 3] or [1, 40, 80, 1]
        input = input.permute(0, 3, 1, 2)           # Torch Tensor: [1, 3, 40, 80] or [1, 1, 40, 80] 
                                                    # PyTorch uses channel first format

        return self.forward(input).squeeze().detach().numpy()