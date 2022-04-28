import os
import torch
import argparse
import numpy as np

from utils.env import launch_env
from utils.teacher import PurePursuitExpert
from utils.loggers import Logger
from utils.wrappers import steering_to_wheel
from utils.wrappers import preprocess_observation, preprocess_observation_GAIL, preprocess_observation_unit
from utils.wrappers import ActionDelayWrapper

from models.model_dpl import PytorchTrainer
from models.model_unit_network import *



# Logging configuration
EPISODES = 128
STEPS = 768

# Variable to set if we want to use Domain Randomization in the Simulator
USE_DOMAIN_RANDOMIZATION = True

# If USE_SAFEDAGGER is False, we will do the classical DAgger algorithm:
# - We will predict the expert's action at every timestep, regardless if the expert or the agent is in control.
# - Therefore, we will log the expert's actions at every step of the environment and use them to teach the agent.
# If USE_SAFEDAGGER is True, we will use the Safe-DAgger algorithm:
# - We will only log the expert's actions if the expert is in control (which happens if the agent is in an usafe state).
USE_SAFEDAGGER = False

# DAgger Distance and angle limits - if the agent is outside the limit, the expert takes control of the robot
DIST_LIMIT = 0.05
ANGLE_LIMIT = 60

# If this variable is True, we want our model to learn/predict the velocity and the steering angle signals (as the actions)
# if it's False,  we want our model to learn/predict the PWM signals of the motors (as actions)
LEARN_STEERING_AND_VELOCITY = True



# Parse arguments - determine which IL algorithm are we going to collect demonstrations for
parser = argparse.ArgumentParser()
parser.add_argument(
    "algorithm",
    choices=["bc", "dagger", "gail_pretrain", "unit_controller"],
    type=str,
    help="Select the IL algorithm",
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

    # Set the UNIT extractor network's weights to non-trainable, so that the gradients of the feature tensors will not cause an error
    for p in E1.parameters():
        p.requires_grad = False

    def unit_observation_preprocesser(observation):
        preprocessed_observation = preprocess_observation_unit(observation)
        _, latent_variable = E1(preprocessed_observation)
        latent_variable = np.array(np.squeeze(latent_variable, axis=0))
        return latent_variable
#-------------------------------------------------------------------------------------------------------------------------------
# Function to perform calculations for the DAgger algorithm
def perform_dagger_calculations(env):
    # Use the get_lane_pos2 function to get the agent's LanePosition variable
    lp = env.env.get_lane_pos2(env.env.cur_pos, env.env.cur_angle)
    # LanePosition.dist is the agent's signed distance from the middle of the right lane
    # LanePosition.angle_deg is the agent's signed angle divergence from the forward direction (measured in degrees)

    # The absolute value of LanePosition.dist is the agent's distance from the middle of the right lane
    # The absolute value of LanePosition.angle_deg is the agent's angle divergence from the forward direction
    unsignedDist = np.absolute(lp.dist)
    unsignedAngle = np.absolute(lp.angle_deg)

    return unsignedAngle, unsignedDist
#-------------------------------------------------------------------------------------------------------------------------------



# Set configuration based on the algorithm
if algo == "bc":
    observation_preprocesser = preprocess_observation
    domain_rand = USE_DOMAIN_RANDOMIZATION
    log_file = '_train.log'
    dagger = False

elif algo == "dagger":
    observation_preprocesser = preprocess_observation
    domain_rand = USE_DOMAIN_RANDOMIZATION
    log_file = '_train.log'
    model = PytorchTrainer().load().eval()
    dagger = True

elif algo == "gail_pretrain":
    observation_preprocesser = preprocess_observation_GAIL
    domain_rand = False
    log_file = '_train.log'
    dagger = False

elif algo == "unit_controller":
    observation_preprocesser = unit_observation_preprocesser
    domain_rand = False
    log_file = '_unit_train.log'
    dagger = False



# Create environment, expert and logger
environment = launch_env(domain_rand=domain_rand, randomize_maps=True, random_seed=573423)
env = ActionDelayWrapper(environment)
expert = PurePursuitExpert(env=env.env, DAgger=dagger)
logger = Logger(log_file=log_file)



# Start collecting demonstrations
observation = env.reset()

for episode in range(0, EPISODES):
    print('Starting episode {}/{}'.format(episode+1, EPISODES))

    for steps in range(0, STEPS):

        # Calculate the agent's distance and angle deviation from the ideal line
        unsignedDist, unsignedAngle = perform_dagger_calculations(env)

        # Preprocess the observation which will be saved later
        preprocessed_observation = observation_preprocesser(observation)


        # Calculate the action that will be used to step the environment

        # Use the expert to control the Duckiebot in any of these cases:
        # - we are not using the DAgger algorithm
        # - the distance is bigger than DIST_LIMIT
        # - the angle divergence is bigger than ANGLE_LIMIT
        if not dagger or unsignedDist > DIST_LIMIT or unsignedAngle > ANGLE_LIMIT:

            # Use the expert to predict the steering angle and velocity
            action = expert.predict(None)

            # The expert provides us with velocity and steering angle, so we need to convert these first into PWM signals
            action = steering_to_wheel(action)

        # Otherwise use the agent for the prediction
        else:

            # Use the model to predict the action
            action = model.predict(preprocessed_observation)

            # If the model provides steering angle and velocity, we need to convert these to PWM signals
            if LEARN_STEERING_AND_VELOCITY:
                action = steering_to_wheel(action)


        # Calculate the expert's action that will be used to teach the agent

        # Use the expert to predict the steering angle and velocity
        expert_action = expert.predict(None)

        # If we want our model to learn PWM signals, we need to convert the expert's signals into PWM signals
        if not LEARN_STEERING_AND_VELOCITY:
            expert_action = steering_to_wheel(expert_action)


        # Step the environment using the actions predicted by the expert or the model
        try:
            observation, reward, done, info = env.step(action)
        except Exception as e:
            print("-----------------------------------------------\nEnvironment reset:")
            print(e)

        closest_point, _ = env.env.closest_curve_point(env.env.cur_pos, env.env.cur_angle)
        if closest_point is None:
            done = True
            logger.clear_logs()
            print("\n\nExpert left the track, logs not saved.\n\n")
            break


        # Do the logging of the data in either of these cases:
        #   - if we are not using the DAgger algorithm, we log in every timestep
        #   - USE_SAFEDAGGER is False, we use the classical DAgger algorithm, therefore we log in every timestep
        #   - USE_SAFEDAGGER is True, and the action was done by the expert (Duckiebot was not within the limits), we log that action
        if not dagger or not USE_SAFEDAGGER or (unsignedDist > DIST_LIMIT) or (unsignedAngle > ANGLE_LIMIT):

            # Log the observation-action pairs
            logger.log(preprocessed_observation, expert_action, reward, done, info)

        env.render() # to watch the expert interaction with the environment

    logger.on_episode_done()  # speed up logging by flushing the file
    observation = env.reset()

# we flush everything and close the file
logger.close()

env.close()

# Make a beep noise to alert that the logging has ended
print('\a')
print('\a')