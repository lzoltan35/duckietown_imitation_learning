import gym
import gym_duckietown

def launch_env(domain_rand, id=None, random_seed=4000, randomize_maps=False, max_steps=500000):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=random_seed, # random seed
            map_name="ETHZ_autolab_technical_track_grass",
            max_steps=max_steps, # we don't want the gym to reset itself
            domain_rand=domain_rand,
            camera_rand=domain_rand,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
            randomize_maps_on_reset=randomize_maps,
            frame_skip=1
        )
    else:
        env = gym.make(id)

    return env
