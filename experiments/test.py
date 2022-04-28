import argparse
import glob
import os
import torch

from utils.env import launch_env
from utils.wrappers import wheel_to_steering
from utils.wrappers import steering_to_wheel
from utils.wrappers import preprocess_observation, preprocess_observation_GAIL, preprocess_observation_unit
from utils.wrappers import ActionDelayWrapper

from models.model_dpl import PytorchTrainer
from models.model_gail import Policy
from models.model_unit_controller import UnitControllerTrainer
from models.model_unit_network import *


# Test settings
EPISODES = 4
STEPS = 2500
PRINT_VELOCITY_INFO = False

# If this variable is True, we want our model to learn/predict the velocity and the steering angle signals (as the actions)
# if it's False,  we want our model to learn/predict the PWM signals of the motors (as actions)
LEARN_STEERING_AND_VELOCITY = True



# Parse arguments - determine which IL algorithm to test
parser = argparse.ArgumentParser()
parser.add_argument(
    "algorithm",
    choices=["bc", "dagger", "gail", "unit_controller"],
    type=str,
    help="Select which IL algorithm to test in the environment",
)
args = parser.parse_args()
algo = args.algorithm


#----------------------------------------------- UNIT Network configuration -------------------------------------------------------
if algo == "unit_controller":
    epoch = 199                 # epoch to start training from
    img_height = 60             # size of image height
    img_width = 80              # size of image width
    img_channels = 3            # number of image channels
    n_downsample = 2            # number downsampling layers in encoder
    n_dim = 64                  # number of filters in first encoder layer

    cuda = False

    model_save_path = os.getcwd() + '/trained_models/unit'

    input_shape = (img_channels, img_height, img_width)

    # Dimensionality (channel-wise) of image embedding
    shared_dim = n_dim * 2 ** n_downsample

    # Initialize generator and discriminator
    shared_E = ResidualBlock(features=shared_dim)
    E1 = Encoder(dim=n_dim, n_downsample=n_downsample, shared_block=shared_E)

    if cuda:
        E1 = E1.cuda()

    if epoch != 0:
        # Load pretrained models
        E1.load_state_dict(torch.load("%s/E1_%d.pth" % (model_save_path, epoch)))
    else:
        # Initialize weights
        E1.apply(weights_init_normal)

    def unit_observation_preprocesser(observation):
        preprocessed_observation = preprocess_observation_unit(observation)
        _, latent_variable = E1(preprocessed_observation)
        return latent_variable
#-------------------------------------------------------------------------------------------------------------------------------


# Create environment
environment = launch_env(
    domain_rand=False, randomize_maps=False, random_seed=5000
)  # domain_rand=False, because we are in the test phase, therefore we don't want domain randomization
env = ActionDelayWrapper(environment)


# Set the model and the observation preprocesser based on the algorithm
if algo == "bc" or algo == "dagger":
    model = PytorchTrainer().load().eval()
    observation_preprocesser = preprocess_observation

elif algo == "gail":
    model = Policy(log_std=-0.0).to(torch.device('cpu'))
    best_model_path = glob.glob("trained_models/gail/policy_net_best_epoch_*.pth")[0]
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    observation_preprocesser = preprocess_observation_GAIL

elif algo == "unit_controller":
    model = UnitControllerTrainer().load().eval()
    observation_preprocesser = unit_observation_preprocesser


# Start testing the algorithm
observation = env.reset()

# We can use the gym reward to get an idea of the performance of our model
cumulative_reward = 0.0

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        
        observation = observation_preprocesser(observation)
        action = model.predict(observation)

        # If the model provides steering angle and velocity, we need to convert these to PWM signals
        if LEARN_STEERING_AND_VELOCITY:
            action = steering_to_wheel(action)

        if PRINT_VELOCITY_INFO:
            print("\n----------------------------------------")
            print("Differential actions: % 6.4f , % 6.4f" %(action[0], action[1]))
            print("Velocity: % 6.4f , Steering angle: % 6.4f" %(wheel_to_steering(action)[0], wheel_to_steering(action)[1]))

        observation, reward, done, info = env.step(action)
         
        cumulative_reward += reward
        if done:
            env.reset()
            print("\n\nFAILURE: Agent failed to finish episode.\n\n")
        env.render()

    env.reset()
    print("\n\nSUCCESS: Agent successfully finished episode.\n\n")

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))

env.close()