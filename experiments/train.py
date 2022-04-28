import argparse
import time
import numpy as np
from tqdm import tqdm

from utils.loggers import Reader
from utils.gail_utils import extract_features_for_gail

from models.model_dpl import PytorchTrainer
from models.model_gail import PolicyPretrainer
from models.model_unit_controller import UnitControllerTrainer

from torch.utils.tensorboard import SummaryWriter


# Training settings
BATCH_SIZE = 32
MAX_EPOCHS = 1500
TRAIN_SPLIT = 0.8
PATIENCE = 25
USE_EARLY_STOPPING = True

SAVE_EPOCHS_REGURARLY = True
EPOCH_SAVING_FREQUENCY = 5


# Parse arguments - determine which algorithm to train
parser = argparse.ArgumentParser()
parser.add_argument(
    "algorithm",
    choices=["bc", "dagger", "gail_pretrain", "unit_controller"],
    type=str,
    help="Select which algorithm to train",
)
args = parser.parse_args()
algo = args.algorithm


# Set configuration based on the algorithm
if algo == "bc":
    model_name = 'BC_{}'.format(int(time.time()))
    log_files = '_train.log'
    model = PytorchTrainer()
    
elif algo == "dagger":
    model_name = 'DAgger_{}'.format(int(time.time()))
    log_files = '_train.log'
    model = PytorchTrainer()

elif algo == "unit_controller":
    model_name = 'UNIT_{}'.format(int(time.time()))
    log_files = '_unit_train.log'
    model = UnitControllerTrainer()

elif algo == 'gail_pretrain':
    model_name = 'GAIL_{}'.format(int(time.time()))
    log_files = '_features.log'
    model = PolicyPretrainer()

    # If we want to pretrain a GAIL network, we need to convert the logs to features first
    extract_features_for_gail()


# Create training data reader and TensorBoard logger
reader = Reader(log_files)
writer = SummaryWriter('tensorboard/{}'.format(model_name))


# Import training data
# model input types can be observations (bc, dagger, image-gail), features (feature-gail) or latent variables (unit)
model_inputs, actions = reader.read()  
actions = np.array(actions)
model_inputs = np.array(model_inputs)

prev_loss = 100.0
best_epoch = 0

print("\n---------------------------------------\n     Number of training data:")
print("     " + format(len(model_inputs)))
if USE_EARLY_STOPPING:
    print("     Training WITH Early Stopping\n---------------------------------------")
else:
    print("     Training WITHOUT Early Stopping\n---------------------------------------")


# Randomly shuffle the whole dataset
data_num = len(model_inputs)
permute = np.random.permutation(data_num)
model_inputs = model_inputs[permute]
actions = actions[permute]

batch_num = data_num // BATCH_SIZE 

if USE_EARLY_STOPPING:

    # Split data into training and validation sets
    model_inputs_train = model_inputs[0:int(data_num*TRAIN_SPLIT)]
    actions_train = actions[0:int(data_num*TRAIN_SPLIT)]
    model_inputs_valid = model_inputs[int(data_num*TRAIN_SPLIT):]
    actions_valid = actions[int(data_num*TRAIN_SPLIT):]

    train_num = len(model_inputs_train)
    valid_num = len(model_inputs_valid)

    train_batch_num = train_num // BATCH_SIZE 
    valid_batch_num = valid_num // BATCH_SIZE 

    epochs_no_improve = 0

epochs_bar = tqdm(range(MAX_EPOCHS))
for epoch in epochs_bar:
    
    if USE_EARLY_STOPPING: 
        
        # Training WITH early stopping
        train_loss = 0.0
        valid_loss = 0.0

        # Randomly shuffle the training data
        train_permute = np.random.permutation(train_num)
        model_inputs_train = model_inputs_train[train_permute]
        actions_train = actions_train[train_permute]

        # Randomly shuffle the validation data
        valid_permute = np.random.permutation(valid_num)
        model_inputs_valid = model_inputs_valid[valid_permute]
        actions_valid = actions_valid[valid_permute]

        # Iterate trough the training data and use them to train the network
        for batch in range(0, train_num, BATCH_SIZE):
            model_input_batch = model_inputs_train[batch:batch + BATCH_SIZE]
            action_batch = actions_train[batch:batch + BATCH_SIZE]
            train_loss += model.train(model_input_batch,action_batch)

        # Iterate trough the validation data and calculate the validation loss
        for batch in range(0, valid_num, BATCH_SIZE):
            model_input_batch = model_inputs_valid[batch:batch + BATCH_SIZE]
            action_batch = actions_valid[batch:batch + BATCH_SIZE]
            valid_loss += model.calculate_validation_loss(model_input_batch,action_batch)

        # Calculate the average of the train and validation losses (w.r.t the batches)
        train_loss /= train_batch_num
        valid_loss /= valid_batch_num

        epochs_bar.set_postfix({'train': train_loss, 'valid': valid_loss})
        writer.add_scalars('loss',{'training':train_loss, 'validation':valid_loss}, epoch)

        if SAVE_EPOCHS_REGURARLY and epoch % EPOCH_SAVING_FREQUENCY == 0:
            model.save_epoch(epoch_number=epoch)

        if(prev_loss > valid_loss):
            model.save()
            prev_loss = valid_loss
            best_epoch = epoch
            epochs_no_improve = 0
            print('\nModel saved...')
        else:
            epochs_no_improve += 1
            print('\nValiadtion error has not improved for ' + format(epochs_no_improve) + ' epochs...')   

        if epochs_no_improve == PATIENCE:
            print('Early stopping...')
            model.rename_best_epoch(epoch_number=best_epoch)
            break
    else:

        # Training WITHOUT early stopping
        loss = 0.0

        # Randomly shuffle the training data
        permute = np.random.permutation(data_num)
        model_inputs = model_inputs[permute]
        actions = actions[permute]

        # Iterate trough the training data and use them to train the network
        for batch in range(0, data_num, BATCH_SIZE):
            model_input_batch = model_inputs[batch:batch + BATCH_SIZE]
            action_batch = actions[batch:batch + BATCH_SIZE]
            loss += model.train(model_input_batch,action_batch)    

        # Calculate the average of the train loss (w.r.t the batches)
        loss /= batch_num

        epochs_bar.set_postfix({'loss': loss})
        writer.add_scalar('loss', loss, epoch)

        if SAVE_EPOCHS_REGURARLY and epoch % EPOCH_SAVING_FREQUENCY == 0:
            model.save_epoch(epoch_number=epoch)

        if(prev_loss > loss):
            model.save()
            prev_loss = loss
            best_epoch = epoch
            print('\nModel saved...')
        else:
            print('\nWorse model, not saved...')

    if epoch == MAX_EPOCHS-1:
        model.rename_best_epoch(epoch_number=best_epoch)


if USE_EARLY_STOPPING:
    print('The final validation loss is: ' + format(prev_loss))
else:
    print('The final loss is: ' + format(prev_loss))

# release the resources
reader.close()
writer.close()

# Make a beep noise to alert that the logging has ended
print('\a')
print('\a')