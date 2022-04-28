import torch 
import torch.nn as nn

import numpy as np
import os
import glob



class UnitControllerTrainer:
    def __init__(self, learning_rate=0.0001):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UnitControllerModel(output_size=2).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        model_save_path = 'trained_models/unit_controller' 
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            print('\nFolder created: ', model_save_path)

    def train(self, latent_variable_batch, action_batch):
         
        latent_variable_batch = torch.Tensor(latent_variable_batch).to(self.device)  
        action_batch = torch.Tensor(action_batch).to(self.device)

        self.optimizer.zero_grad()
        output = self.model(latent_variable_batch)
        loss = self.criterion(output.squeeze(), action_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_validation_loss(self, latent_variable_batch, action_batch):
        
        latent_variable_batch = torch.Tensor(latent_variable_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)

        with torch.no_grad():
            
            output = self.model(latent_variable_batch)
            loss = self.criterion(output.squeeze(), action_batch)

        return loss.item()

    def save(self):
        torch.save(self.model.state_dict(), 'trained_models/unit_controller/unit_ctrl_convnet.pth')
    
    def save_epoch(self, epoch_number):
        path = 'trained_models/unit_controller/unit_ctrl_convnet_epoch_{}.pth'.format(epoch_number)
        torch.save(self.model.state_dict(), path)

    def rename_best_epoch(self, epoch_number):
        old_name = 'trained_models/unit_controller/unit_ctrl_convnet.pth'
        new_name = 'trained_models/unit_controller/unit_ctrl_convnet_best_epoch_{}.pth'.format(epoch_number)
        os.rename(old_name, new_name)

    def load(self):
        model_path = "trained_models/unit_controller/unit_ctrl_convnet_best_epoch_*.pth"
        best_model_path = glob.glob(model_path)[0]
        self.model.to(torch.device('cpu'))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        return self.model



class UnitControllerModel(nn.Module):
    def __init__(self, output_size=2):
        super(UnitControllerModel, self).__init__()

        # Input: 12x20x256 ( H x W x CH )
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Output: 6x10x512 ( H/2 x W/2 x 512 )

        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 768, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Output: 3x5x768 ( H/4 x W/4 x 768 )

        self.fc = nn.Linear(768 * 3 * 5, output_size)
        
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, input): 

        return self.forward(input).squeeze().detach().numpy()