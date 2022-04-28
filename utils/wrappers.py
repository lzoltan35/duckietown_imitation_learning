import gym
from gym import spaces
from gym_duckietown.simulator import Simulator
import numpy as np
import cv2
import torch
from torchvision.transforms import ToTensor, Normalize, Compose

# If this variable is true, the image preprocessing wrapper will not only resize and crop the image, but it will also perform
# a thresholding operation on the image to extract the white and yellow lines
IMAGE_THRESHOLDING = False

# If this variable is true, the image preprocessing wrapper will normalize the image
USE_NORMALIZATION = True

# Create normalizing transformations
gail_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unit_normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

class ActionDelayWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ActionDelayWrapper, self).__init__(env)
        self.simulator = self.unwrapped  # type: Simulator
        assert isinstance(self.simulator, Simulator), "Env must be gym_duckietown.simulator.Simulator"
        self.action_delay_ratio = 0.25
        assert self.action_delay_ratio > 0.0 and self.action_delay_ratio < 1.0, "action_delay_ratio must be in the (0, 1) interval"
        self.last_action = np.zeros(self.action_space.shape)

    def step(self, action):
        delta_time = 1.0 / self.simulator.frame_rate
        # Apply the action delay
        for _ in range(self.simulator.frame_skip):
            self.simulator.update_physics(action=self.last_action, delta_time=delta_time * self.action_delay_ratio)
            # Update physics increments step count but that should ony be incremented once for each step
            # This will happen when self.env.step() is called
            self.simulator.step_count -= 1
        self.last_action = action
        self.simulator.delta_time = delta_time * (1. - self.action_delay_ratio)
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = np.zeros(self.action_space.shape)
        return self.env.reset(**kwargs)


# Converts velocity|steering actions to differential wheel actions: wheel_left|wheel_right
def steering_to_wheel(action):
    gain=1.0
    trim=0.0
    radius=0.0318
    k=27.0
    limit=1.0
    wheel_dist=0.102
    vel, angle = action

    # assuming same motor constants k for both motors
    k_r = k
    k_l = k

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel + 0.5 * angle * wheel_dist) / radius
    omega_l = (vel - 0.5 * angle * wheel_dist) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, limit), -limit)
    u_l_limited = max(min(u_l, limit), -limit)

    vels = np.array([u_l_limited, u_r_limited])
    return vels


# Converts wheel_left|wheel_right differential wheel actions to velocity|steering actions
def wheel_to_steering(action):
    gain=1.0
    trim=0.0
    radius=0.0318
    k=27.0
    limit=1.0
    wheel_dist=0.102

    v_l, v_r = action

    velocity = (v_l + v_r)/2
    angle = (v_r - v_l)/radius/8 
    #NOTE: This constant (8) might not be the proper constant. I have no mathematical reason behind using it, 
    # I just tried a few constants and this seemed to work fine. (If we dont use this, we get huge values for the steering) 

    return velocity, angle


def preprocess_observation(observation): 

    # Resize the observation to 60x80
    observation = cv2.resize(observation, (80, 60))
    # Crop the image to 40x80 (remove the upper part)
    observation = observation[20:60, :]

    if IMAGE_THRESHOLDING:
        # Convert the image to the HSV colour space
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)

        # Apply thresholding to the image to extract the yellow and white lines
        mask1 = cv2.inRange(observation, (0, 60, 90), (179, 255, 255))  # Mask for the yellow lines | old values: (70,50,150), (120, 255, 255)
        mask2 = cv2.inRange(observation, (0, 0, 70), (179, 60, 255))    # Mask for the white lines  | old values: (0,0,140), (255, 120, 255)    | older values: (0,0,140), (255, 80, 255)

        # Combine the masks
        observation = cv2.bitwise_or(mask1, mask2)

        # Change image dimension from (40, 80) to (40, 80, 1) to match the 3 channel training format
        observation = np.expand_dims(observation, axis=2)
    else:
        # Change the order of the channels because of OpenCV
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

    return observation


def preprocess_observation_real(observation): 

    if IMAGE_THRESHOLDING:
        # Convert the image to the HSV colour space
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)

        # Apply thresholding to the image to extract the yellow and white lines
        mask1 = cv2.inRange(observation, (25, 60, 140), (35, 255, 255))
        mask2 = cv2.inRange(observation, (0, 0, 165), (179, 60, 255))

        # Combine the masks
        observation = cv2.bitwise_or(mask1, mask2)

        # Change image dimension from (40, 80) to (40, 80, 1) to match the 3 channel training format
        observation = np.expand_dims(observation, axis=2)
    else:
        # Change the order of the channels because of OpenCV
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

    # Resize the observation to 60x80
    observation = cv2.resize(observation, (80, 60))
    # Crop the image to 40x80 (remove the upper part)
    observation = observation[20:60, :]

    return observation


def preprocess_observation_GAIL(observation): 

    # Resize the observation to 60x80
    observation = cv2.resize(observation, (80, 60))
    # Crop the image to 40x80 (remove the upper part)
    observation = observation[20:60, :]

    # Change the order of the channels because of OpenCV
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

    if USE_NORMALIZATION:
        observation = torch.tensor(observation).float().permute(2, 0, 1)
        observation = gail_normalize(observation).permute(1, 2, 0).numpy()

    return observation    


def preprocess_observation_unit(observation): 

    # Resize the observation to 60x80
    observation = cv2.resize(observation, (80, 60))

    # Change the order of the channels because of OpenCV
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

    observation = torch.tensor(observation).float().permute(2, 0, 1)
    observation = unit_normalize(observation).unsqueeze(0)

    return observation   


def preprocess_observation_dagger_baseline(observation):

        input_shape=(60, 80)

        # Resize images
        observation = cv2.resize(observation, dsize=input_shape[::-1])
        # Transform to tensors
        compose_obs = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # using imagenet normalization values
        ])
        
        observation = compose_obs(observation).unsqueeze(0)

        return observation