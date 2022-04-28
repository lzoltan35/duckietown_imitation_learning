import argparse
import torch
import glob

from utils.env import launch_env
from utils.wrappers import ActionDelayWrapper

from models.model_dpl import PytorchTrainer
from models.squeezenet import Squeezenet
from models.model_gail import Policy

from utils.duckietown_world_evaluator import DuckietownWorldEvaluator


# Evaluator settings
DEFAULT_FRAMERATE = 30
EVAL_LENGTH_SEC = 15

# An official evaluation episode is 15 seconds long
episode_max_steps = EVAL_LENGTH_SEC * DEFAULT_FRAMERATE


# Parse arguments - determine which IL algorithm to evaluate
parser = argparse.ArgumentParser()
parser.add_argument(
    "algorithm",
    choices=["bc", "dagger", "gail", "baseline_dagger"],
    type=str,
    help="Select which IL algorithm to evaluate",
)
args = parser.parse_args()
algo = args.algorithm


# Set configuration based on the IL algorithm
if algo == "bc" or algo == "dagger":
    use_gail = False
    use_dagger_baseline = False
    eval_path = "./evaluation_results/dpl"
elif algo == "baseline_dagger":
    use_gail = False
    use_dagger_baseline = True
    eval_path = "./evaluation_results/dagger_baseline"
elif algo == "gail":
    use_gail = True
    use_dagger_baseline = False
    eval_path = "./evaluation_results/gail"


# Create environment
environment = launch_env(
    domain_rand=False, randomize_maps=False, random_seed=13, max_steps=episode_max_steps
)  # domain_rand=False, because we are in the evaluation phase, therefore we don't want domain randomization
env = ActionDelayWrapper(environment)


# Select which model to use (based on the IL algorithm):
if algo == "bc" or algo == "dagger":
    model = PytorchTrainer().load().eval()

elif algo == "baseline_dagger":
    model = Squeezenet(num_outputs=2, max_velocity=0.5)
    model.load_state_dict(torch.load("trained_models/dagger_baseline/model.pt", map_location="cpu"))

elif algo == "gail":
    model = Policy(log_std=-0.0).to(torch.device('cpu'))
    best_model_path = glob.glob("trained_models/gail/policy_net_best_epoch_*.pth")[0]
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))


# Evaluate the model
evaluator = DuckietownWorldEvaluator(
    env=env, use_gail=use_gail, use_dagger_baseline=use_dagger_baseline
)
evaluator.evaluate(model, eval_path)

print("Evaluation finished!")

env.close()
