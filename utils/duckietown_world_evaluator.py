"""
Copyright Notice:
    This evaluator script was created based on the implementation of András Kalapos.
    The original version can be found here: 
    https://github.com/kaland313/Duckietown-RL/blob/master/duckietown_utils/duckietown_world_evaluator.py


DuckietownWorldEvaluator Evaluates an RLlib reinforcement learning agent using the same evaluator (duckietown-world),
which is used in DTS evaluate.
Duckietown World:
   takes trajectory data, which is recorded on one of it's "built in maps".
   https://github.com/duckietown/duckietown-world
   https://pypi.org/project/duckietown-world-daffy
   Supported version: 5.0.11 (designed for this, but may work with others)
DuckietownWorldEvaluator orchestrates the trajectory recording and evaluation for an RLlib agent.
Usage example:
   evaluator = DuckietownWorldEvaluator(config['env_config'])
   evaluator.evaluate(trainer, './EvaluationResults')

To adapt for a different agent implementation than RLlib, DuckietownWorldEvaluator should be subclassed and
   __init__ and _compute_action, (_collect_trajectory) should be overrided (modified)
Custom maps should be copied to the installation folder of Duckietown Worldlib .../duckietown_world/data/gd1/maps
   e.g. /home/username/miniconda3/envs/duckietownthesis/lib/python3.6/site-packages/duckietown_world/data/gd1/maps
"""
__author__= "András Kalapos"
__license__= "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
import numpy as np
import json
import time
import logging
import copy

from duckietown_world import SE2Transform
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import EvaluatedMetric, make_timeseries
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing import draw_static
from duckietown_world.world_duckietown.duckiebot import DB18
from duckietown_world.world_duckietown.map_loading import load_map

from utils.wrappers import steering_to_wheel
from utils.wrappers import preprocess_observation, preprocess_observation_GAIL, preprocess_observation_dagger_baseline

DEFAULT_EVALUATION_MAP = 'ETHZ_autolab_technical_track'
logger = logging.getLogger(__name__)

# If this variable is True, we want our model to learn/predict the velocity and the steering angle signals (as the actions)
# if it's False,  we want our model to learn/predict the PWM signals of the motors (as actions)
LEARN_STEERING_AND_VELOCITY = True

class DuckietownWorldEvaluator:
    """
    Evaluates a RLlib agent using the same evaluator which is used in DTS evaluate.
    To adapt for a different agent implementation than RLlib, __init__ and _compute_action,
    (_collect_trajectory) should be modified.
    """
    # These start poses are exactly the same as used by dts evaluate
    # Pose description [[x, 0, z], rot].
    # x, z are measured in meters (x: horizontal, z: vertical, [0,0,0]: top left corner), rot is measured in radians
    start_poses = {'ETHZ_autolab_technical_track': [[[0.7019999027252197, 0, 0.41029359288296474], -0.2687807048071267],
                                                    [[0.44714101540138385, 0, 2.2230001401901243], 1.31423292675173],
                                                    [[1.5552862449923595, 0, 1.0529999446868894], 1.4503686084072878],
                                                    [[1.6380000114440918, 0, 3.0929162652880935], -3.0264009229581674],
                                                    [[0.3978251477698904, 0, 2.8080000591278074], 1.2426744274199626],
                                                    # [[0.2, 0, 2.8], np.pi/2*1.1],  # For testing lane-correction
                                                    # [[0.585, 0, 5.75*0.585], np.pi],  # For testing lane-correction
                                                    ]
                   }

    def __init__(self, env, use_gail, use_dagger_baseline=False, eval_lenght_sec=15, eval_map=DEFAULT_EVALUATION_MAP):

        # Set the map
        self.map_name = eval_map
        # Make testing env
        self.env = env

        self.use_gail = use_gail
        self.use_dagger_baseline = use_dagger_baseline

        # Set up evaluator
        # Creates an object 'duckiebot'
        self.ego_name = 'duckiebot'
        self.db = DB18()  # class that gives the appearance
        # load one of the maps
        self.dw = load_map(self.map_name)

    def evaluate(self, agent, outdir, episodes=None):
        """
        Evaluates the agent on the map inicialised in __init__
        :param agent: Agent to be evaluated, passed to self._collect_trajectory(agent,...)
        :param outdir: Directory for logged outputs (trajectory plots + numeric data)
        :param episodes: Number of evaluation episodes, if None, it is determined based on self.start_poses
        """
        if episodes is None:
            episodes = len(self.start_poses.get(self.map_name, []))
        totals = {}
        for i in range(episodes):
            episode_path, episode_orientations, episode_timestamps = self._collect_trajectory(agent, i)
            logger.info("Episode {}/{} sampling completed".format(i+1, episodes))
            if len(episode_timestamps) <= 1:
                continue
            episode_path = np.stack(episode_path)
            episode_orientations = np.stack(episode_orientations)
            # Convert them to SampledSequences
            transforms_sequence = []
            for j in range(len(episode_path)):
                transforms_sequence.append(SE2Transform(episode_path[j], episode_orientations[j]))
            transforms_sequence = SampledSequence.from_iterator(enumerate(transforms_sequence))
            transforms_sequence.timestamps = episode_timestamps

            _outdir = outdir
            if outdir is not None and episodes > 1:
                _outdir = os.path.join(outdir, "Trajectory_{}".format(i+1))
            evaluated = self._eval_poses_sequence(transforms_sequence, outdir=_outdir)
            logger.info("Episode {}/{} plotting completed".format(i+1, episodes))
            totals = self._extract_total_episode_eval_metrics(evaluated, totals, display_outputs=True)

        # Calculate the median total metrics
        median_totals = {}
        mean_totals = {}
        stdev_totals = {}
        for key, value in totals.items():
            median_totals[key] = np.median(value)
            mean_totals[key] = np.mean(value)
            stdev_totals[key] = np.std(value)
        # Save results to file
        with open(os.path.join(outdir, "total_metrics.json"), "w") as json_file:
            json.dump({'median_totals': median_totals,
                       'mean_totals': mean_totals,
                       'stdev_totals': stdev_totals,
                       'episode_totals': totals}, json_file, indent=2)

        logger.info("\nMedian total metrics: \n {}".format((median_totals)))
        logger.info("\nMean total metrics: \n {}".format((mean_totals)))
        logger.info("\nStandard deviation of total metrics: \n {}".format((stdev_totals)))

    def _collect_trajectory(self, agent, i):
        episode_path = []
        episode_orientations = []
        episode_timestamps = []
        if self.map_name in self.start_poses.keys():
            self.env.unwrapped.user_tile_start = [0, 0]
            self.env.unwrapped.start_pose = self.start_poses[self.map_name][i]
        obs = self.env.reset()
        done = False
        while not done:
            action = self._compute_action(agent, obs)
            obs, reward, done, info = self.env.step(action)
            cur_pos = correct_gym_duckietown_coordinates(self.env.unwrapped, self.env.unwrapped.cur_pos)
            episode_path.append(cur_pos)
            episode_orientations.append(np.array(self.env.unwrapped.cur_angle))
            episode_timestamps.append(info['Simulator']['timestamp'])
        self.env.unwrapped.start_pose = None
        self.user_tile_start = None
        return episode_path, episode_orientations, episode_timestamps

    def _compute_action(self, agent, obs):
        """
        This function should be modified for other agents!
        :param agent: Agent to be evaluated.
        :param obs: New observation
        :return: Action computed based on action
        """
        if self.use_dagger_baseline:
            obs = preprocess_observation_dagger_baseline(obs)
        elif self.use_gail:
            obs = preprocess_observation_GAIL(obs)
        else:
            obs = preprocess_observation(obs)
        action = agent.predict(obs)

        # If the model provides steering angle and velocity, we need to convert these to PWM signals
        if LEARN_STEERING_AND_VELOCITY:
            action = steering_to_wheel(action)

        return action

    def _eval_poses_sequence(self, poses_sequence, outdir=None):
        """
        :param poses_sequence:
        :param outdir: If None evaluation outputs plots won't be saved
        :return:
        """
        # puts the object in the world with a certain "ground_truth" constraint
        self.dw.set_object(self.ego_name, self.db, ground_truth=poses_sequence)
        # Rule evaluation (do not touch)
        interval = SampledSequence.from_iterator(enumerate(poses_sequence.timestamps))
        evaluated = evaluate_rules(poses_sequence=poses_sequence,
                                   interval=interval, world=self.dw, ego_name=self.ego_name)
        if outdir is not None:
            timeseries = make_timeseries(evaluated)
            draw_static(self.dw, outdir, timeseries=timeseries)
        #print(self.dw.get_drawing_children())
        self.dw.remove_object(self.ego_name)
        self.dw.remove_object('visualization')
        return evaluated

    @staticmethod
    def _extract_total_episode_eval_metrics(evaluated, totals, display_outputs=False):
        episode_totals = {}
        for k, rer in evaluated.items():
            from duckietown_world.rules import RuleEvaluationResult
            assert isinstance(rer, RuleEvaluationResult)
            for km, evaluated_metric in rer.metrics.items():
                assert isinstance(evaluated_metric, EvaluatedMetric)
                episode_totals[k] = evaluated_metric.total
                if not (k in totals):
                    totals[k] = [evaluated_metric.total]
                else:
                    totals[k].append(evaluated_metric.total)
        if display_outputs:
            logger.info("\nEpisode total metrics: \n {}".format((episode_totals)))

        return totals

def correct_gym_duckietown_coordinates(sim, pos):
    """
    Gym duckietown uses a weird, coordinate system where the pos[2], z direction is flipped, and shifted.
    Duckietown world usese the unflipped version, this function handles this correction.
    The calculation is based on gym_duckietown.simulator.Simulator.cartesian_from_weird(pos, angle)
    :param sim: gym_duckietown.simulator.Simulator
    :param pos:
    :return:
    """
    # cartesian_position_SE2 = sim.cartesian_from_weird(pos, angle)
    # transform = cartesian_position_SE2.transform_values(SE2Transform.from_SE2)  #type: SE2Transform
    return np.array([pos[0], sim.grid_height * sim.road_tile_size - pos[2]])