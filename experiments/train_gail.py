"""
Copyright Notice:
    The implementation of our GAIL network was created based on the following public repository:
    https://github.com/Khrylx/PyTorch-RL
"""

import os
import glob
import numpy as np
import math
from tqdm import tqdm
from utils.loggers import Reader, ReplayBuffer
from utils.wrappers import ActionDelayWrapper
from utils.env import launch_env

import torch 
import torch.nn as nn

from models.model_gail import Policy
from models.model_gail import Value
from models.model_gail import Discriminator
from utils.gail_utils import ppo_step, estimate_advantages, to_device, roll_out_policy

# configuration zone
BATCH_SIZE = 32
MAX_EPOCHS = 30
TRAJECTORIES_PER_ROLLOUT = 15
STEPS_PER_TRAJECTORIES = 256
REPLAY_BUFFER_SIZE = 75 # 153600 if we store individual steps, 75 if we store trajectories 
SAMPLE_SIZE = 50

# optimization epoch number and batch size for PPO
OPTIM_EPOCHS = 10
OPTIM_BATCH_SIZE = 64

LOG_STD = -0.0
GAMMA = 0.99
TAU = 0.9
L2_REG = 1e-3
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2

SHOW_ROLLOUTS = True
LEARN_STEERING_AND_VELOCITY = True

LOAD_PRETRAINED_GENERATOR = True

SAVE_EPOCHS_REGURARLY = False
EPOCH_SAVING_FREQUENCY = 1

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Creating environment
environment = launch_env(domain_rand=True, randomize_maps=True, random_seed=5709)
env = ActionDelayWrapper(environment)

# Creating model folders:
model_save_path = 'trained_models/gail' 
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    print('\nFolder created: ', model_save_path)

# Creating models
value_net = Value()
discriminator_net = Discriminator()
policy_net = Policy(log_std=LOG_STD)
if LOAD_PRETRAINED_GENERATOR:
    best_model_path = glob.glob("trained_models/gail/policy_net_best_epoch_*.pth")[0]
    policy_net.load_state_dict(torch.load(best_model_path, map_location='cpu'))

# Criterions
discriminator_criterion = nn.BCELoss()
to_device(device, policy_net, value_net, discriminator_net)

# Optimizers
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
optimizer_discriminator = torch.optim.Adam(discriminator_net.parameters(), lr=LEARNING_RATE)

# Initialize Reader and Replay buffer
reader = Reader('_train.log')
buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

# Load expert demonstrations (observation-action pairs)
exp_observations, exp_actions = reader.read()
exp_actions = np.array(exp_actions)
exp_observations = np.array(exp_observations)

print("\n---------------------------------------\n     Number of expert demonstation:")
print("     " + format(len(exp_observations)))
print('\n     Starting training\n\n---------------------------------------')



epochs_bar = tqdm(range(MAX_EPOCHS))
for epoch in epochs_bar:
    
    policy_net.to('cpu')

    # Roll out the current policy in the environment
    trajectories = roll_out_policy(
        agent=policy_net,
        env=env,
        trajectories_per_rollout=TRAJECTORIES_PER_ROLLOUT,
        steps_per_trajectories=STEPS_PER_TRAJECTORIES,
        device="cpu",
        learn_steering_angle_and_velocity=LEARN_STEERING_AND_VELOCITY,
        show_rollouts=SHOW_ROLLOUTS,
    )

    policy_net.to(device=device)

    # Update the replay buffer
    for trajectory in trajectories:
        buffer.add(trajectory)

    # Get samples of the agent's trajectories from the replay buffer
    samples = buffer.get_sample(SAMPLE_SIZE)

    agnt_observations = np.concatenate([sample['observations'] for sample in samples])
    agnt_actions = np.concatenate([sample['actions'] for sample in samples])
    agnt_rewards = np.concatenate([sample['rewards'] for sample in samples])

    # Randomly shuffle the expert's dataset
    exp_data_num = len(exp_observations)
    permute = np.random.permutation(exp_data_num)
    exp_observations = exp_observations[permute]
    exp_actions = exp_actions[permute]

    # Randomly shuffle the agent's dataset
    agnt_data_num = len(agnt_observations)
    permute = np.random.permutation(agnt_data_num)
    agnt_observations = agnt_observations[permute]
    agnt_actions = agnt_actions[permute]
    agnt_rewards = agnt_rewards[permute]

    # Get the number of training data
    data_num = min(agnt_data_num, exp_data_num)

    # Iterate trough the training data and use them to train the networks
    for batch in range(0, data_num, BATCH_SIZE):

        # Create mini-batches
        agnt_observation_batch = agnt_observations[batch:batch + BATCH_SIZE]
        agnt_action_batch = agnt_actions[batch:batch + BATCH_SIZE]
        agnt_reward_batch = agnt_rewards[batch:batch + BATCH_SIZE]
        exp_observation_batch = exp_observations[batch:batch + BATCH_SIZE]
        exp_action_batch = exp_actions[batch:batch + BATCH_SIZE]

        agnt_observation_batch = torch.Tensor(agnt_observation_batch).float().to(device).permute(0, 3, 1, 2)
        agnt_action_batch = torch.Tensor(agnt_action_batch).float().to(device)
        agnt_reward_batch = torch.Tensor(agnt_reward_batch).float().to(device)
        exp_observation_batch = torch.Tensor(exp_observation_batch).float().to(device).permute(0, 3, 1, 2)
        exp_action_batch = torch.Tensor(exp_action_batch).float().to(device)

        with torch.no_grad():
            values = value_net(agnt_observation_batch)
            fixed_log_probs = policy_net.get_log_prob(agnt_observation_batch, agnt_action_batch)

        # Get advantage estimation from the trajectories
        advantages, returns = estimate_advantages(agnt_reward_batch, values, GAMMA, TAU, device)

        # Update discriminator
        for _ in range(1):
            g_o = discriminator_net(agnt_observation_batch, agnt_action_batch)
            e_o = discriminator_net(exp_observation_batch, exp_action_batch)
            optimizer_discriminator.zero_grad()
            discriminator_loss = discriminator_criterion(g_o, torch.ones((agnt_observation_batch.shape[0], 1), device=device)) + \
                discriminator_criterion(e_o, torch.zeros((exp_observation_batch.shape[0], 1), device=device))
            discriminator_loss.backward()
            optimizer_discriminator.step()

        # Perform mini-batch PPO update
        optim_iter_num = int(math.ceil(agnt_observation_batch.shape[0] / OPTIM_BATCH_SIZE))
        for _ in range(OPTIM_EPOCHS):
            perm = np.arange(agnt_observation_batch.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)

            observations, actions, returns, advantages, fixed_log_probs = \
                agnt_observation_batch[perm].clone(), agnt_action_batch[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * OPTIM_BATCH_SIZE, min((i + 1) * OPTIM_BATCH_SIZE, observations.shape[0]))
                observations_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    observations[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, observations_b, actions_b, returns_b,
                        advantages_b, fixed_log_probs_b, CLIP_EPSILON, L2_REG)


    if SAVE_EPOCHS_REGURARLY and epoch % EPOCH_SAVING_FREQUENCY == 0:
        torch.save(policy_net.state_dict(), 'trained_models/gail/policy_net_epoch_{}.pth'.format(epoch))
        torch.save(discriminator_net.state_dict(), 'trained_models/gail/discriminator_net_epoch_{}.pth'.format(epoch))
        torch.save(value_net.state_dict(), 'trained_models/gail/value_net_epoch_{}.pth'.format(epoch))

    if epoch == MAX_EPOCHS-1:
        torch.save(policy_net.state_dict(), 'trained_models/gail/policy_net_best_epoch_x.pth')
        torch.save(discriminator_net.state_dict(), 'trained_models/gail/discriminator_net.pth')
        torch.save(value_net.state_dict(), 'trained_models/gail/value_net.pth')

    torch.cuda.empty_cache()


# release the resources
reader.close()
env.close()

# Make a beep noise to alert that the logging has ended
print('\a')
print('\a')
