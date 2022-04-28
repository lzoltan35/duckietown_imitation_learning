import os
import cv2
import random
import shutil
import glob
import wget
from tqdm import tqdm

from utils.env import launch_env
from utils.teacher import PurePursuitExpert
from utils.wrappers import steering_to_wheel
from utils.wrappers import ActionDelayWrapper


# UNIT Network training data configuration
NUMBER_OF_SIM_IMAGES = 32768    # Number of simulator images (these images will be sampled)
NUMBER_OF_REAL_IMAGES = 30037   # Number of real images (these images will be sampled)
NUMBER_OF_ALL_DATA = 1280
NUMBER_OF_TRAINING_DATA = 1024


# Simulator logging configuration
EPISODES = 64
STEPS = 512

environment = launch_env(domain_rand=False, randomize_maps=True, random_seed=5678)
env = ActionDelayWrapper(environment)
expert = PurePursuitExpert(env=env.env)


# Dataset directories
sim_data_directory = "./unit_data/simData"
real_data_directory = "./unit_data/realData"
target_dataset_directory = "./unit_data/sim2real"
target_dataset_folders = [
    "./unit_data/sim2real/train/A",
    "./unit_data/sim2real/train/B",
    "./unit_data/sim2real/test/A",
    "./unit_data/sim2real/test/B",
]

print('\nCreating dataset directories...\n')
os.makedirs(sim_data_directory, exist_ok=True)
os.makedirs(real_data_directory, exist_ok=True)
for folder in target_dataset_folders:
    os.makedirs(folder, exist_ok=True)


#----------------------------------- Sim Dataset generator -----------------------------------

def generate_sim_dataset(directory):

    print('\nGenerating sim dataset for UNIT network training...\n')
    i=0
    
    for episode in range(0, EPISODES):
        print('Starting episode {}/{}'.format(episode, EPISODES))

        for steps in range(0, STEPS):

            # We use our 'expert' to predict the next action: velocity and steering angle
            action = expert.predict(None)

            # Step the environment using the actions predicted by the expert
            observation, reward, done, info = env.step(steering_to_wheel(action))

            # Save observation image
            image = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            filename = '{}/{}.png'.format(directory,i)
            cv2.imwrite(filename, image)
            i+=1

            env.render() # to watch the expert interaction with the environment
        env.reset()
    env.close()

#----------------------------------- Real Dataset generator -----------------------------------
urls = list()

def download_videos(download_path):
    print("Downloading videos...")
    for i in tqdm(range(len(urls))):
        file_path = os.path.join(download_path, f'{i:03d}.mp4')
        try:
            wget.download(urls[i], file_path)
        except:
            print(f'Download failed for video {urls[i]}')
    print("Downloading finished.")


def saveAsImages(save_path):
    files = sorted(glob.glob(os.path.join(save_path, '*.mp4')))

    print("Saving as .png images...")
    saved_ims = 0
    for file in tqdm(files):
        cap = cv2.VideoCapture(file)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                filename = os.path.join(save_path, f"{saved_ims:06d}.png")
                cv2.imwrite(filename, frame)
                saved_ims += 1
            else:
                break

        cap.release()
        os.remove(file)
    print("Saving finished.")


def generate_real_dataset(directory):
    # Download real images from videos
    print('\nDownloading real dataset for UNIT network training...\n')
    scriptDir = os.path.split(__file__)[0]
    urlFile = os.path.join(scriptDir, "real_video_urls.txt")
    with open(urlFile, 'r') as file:
        for url in file:
            if len(url) > 0:
                urls.append(url.rstrip())

    os.makedirs(directory, exist_ok=True)
    download_videos(directory)
    print('\nConverting videos to images...\n')
    saveAsImages(directory)

    print('\nRenaming images...\n')
    # Rename images to indices
    i=0
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            old_name = '{}/{}'.format(directory,filename)
            new_name = '{}/{}.png'.format(directory,i)
            os.rename(old_name, new_name)
            i+=1
    print('Real dataset created!\n')
#-----------------------------------------------------------------------------------------------


# Generating training datasets
generate_sim_dataset(sim_data_directory)
generate_real_dataset(real_data_directory)

sim_indices = random.sample(range(0, NUMBER_OF_SIM_IMAGES), NUMBER_OF_ALL_DATA)
real_indices = random.sample(range(0, NUMBER_OF_REAL_IMAGES), NUMBER_OF_ALL_DATA)

print('Copying simulator images...')
for counter, index in enumerate(sim_indices):
    original = '{}/{}.png'.format(sim_data_directory,index)
    if counter < NUMBER_OF_TRAINING_DATA:
        target = '{}/train/A/{}.png'.format(target_dataset_directory,index)
    else:
        target = '{}/test/A/{}.png'.format(target_dataset_directory,index)
    shutil.copyfile(original, target)

print('Copying real images...')
for counter, index in enumerate(real_indices):
    original = '{}/{}.png'.format(real_data_directory,index)
    if counter < NUMBER_OF_TRAINING_DATA:
        target = '{}/train/B/{}.png'.format(target_dataset_directory,index)
    else:
        target = '{}/test/B/{}.png'.format(target_dataset_directory,index)
    shutil.copyfile(original, target)

print('\nFinished...')