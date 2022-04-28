"""
Copyright Notice:
    The implementation of our GAIL network was created based on the following public repository:
    https://github.com/Khrylx/PyTorch-RL
"""

import torch 
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from utils.loggers import Reader
from utils.loggers import Logger

from utils.wrappers import preprocess_observation_GAIL, steering_to_wheel


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg):

    #Update critic
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # Weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    # Update policy
    log_probs = policy_net.get_log_prob(states, actions)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()


def estimate_advantages(rewards, values, gamma, tau, device):
    rewards, values = to_device(torch.device('cpu'), rewards, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def to_device(device, *args):
    return [x.to(device) for x in args]


def roll_out_policy(agent, env, trajectories_per_rollout, steps_per_trajectories, device, learn_steering_angle_and_velocity, show_rollouts):

    trajectories = []
    observation = env.reset()
    reward = 0

    for trajectory in range(0, trajectories_per_rollout):

        actions = []
        observations = []
        rewards = []

        for step in range(0, steps_per_trajectories):

            observation = preprocess_observation_GAIL(observation)
            action = agent.predict(observation)

            # Increase reward in every step to encourage agent to do longer rollouts
            reward += 1

            # When the model is not well trained, it sometimes predicts NaN actions. In this case, we reset the environment
            if np.isnan(action).any():
                # current_trajectory = {'actions': np.array(actions), 'observations': np.array(observations), 'rewards': np.array(rewards)}
                # trajectories.append(current_trajectory)
                env.reset()
                print('Model predicted NaN values, restarting...')
                break

            # Add the newly acquired observation-action-reward pairs to the list
            # We need to do this here, before the actions might be converted to PWM signals
            actions.append(action)
            observations.append(observation)
            rewards.append(reward)

            # If the model provides steering angle and velocity, we need to convert these to PWM signals
            if learn_steering_angle_and_velocity:
                action = steering_to_wheel(action)

            observation, _ , done, _ = env.step(action)

            if show_rollouts:
                env.render()

            # If the episode ended (by error or by reaching max steps)
            if done or step == steps_per_trajectories-1:
                current_trajectory = {'actions': np.array(actions), 'observations': np.array(observations), 'rewards': np.array(rewards)}
                trajectories.append(current_trajectory)
                env.reset()
                break
    
    return trajectories


def extract_features_for_gail():

    reader = Reader('_train.log')
    logger = Logger(log_file='_features.log')

    SAVE_INTERVAL_SIZE = 512

    observations, actions = reader.read()
    actions = np.array(actions)
    observations = np.array(observations)
    preprocessed_obs = torch.Tensor(observations).float().permute(0, 3, 1, 2)

    # Creating feature extractor model
    resnet = models.resnet50(pretrained=True).eval()
    modules = list(resnet.children())[:-4]      # Remove the last 2 layers (FC and AveragePooling) and the last ResNet 2 blocks
    feature_extractor = nn.Sequential(*modules)
    for p in feature_extractor.parameters():    # Set the feature extractor network's weights to non-trainable, so that the
        p.requires_grad = False                 # gradients of the feature tensors will not cause an error

    print('Extracting image features...')

    data_num = len(observations)
    epochs_bar = tqdm(range(data_num))

    for i in epochs_bar:

        observation = preprocessed_obs[i]

        feature = np.array(np.squeeze(feature_extractor(observation.unsqueeze(0)), axis=0))
        action = actions[i]
        logger.log(feature, action, 0, 0, 0)

        if i % SAVE_INTERVAL_SIZE == 0:
            logger.on_episode_done()

    logger.on_episode_done()

    print('Image features extracted')

    # release the resources
    reader.close()
    logger.close()

    # Make a beep noise to alert that the logging has ended
    print('\a')
    print('\a')