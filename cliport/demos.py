"""Data collection script."""
import logging
import os
import sys

import hydra
import numpy as np
import random

from loguru import logger
from matplotlib import pyplot as plt

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
from cliport.utils.deferred import deferred
from cliport.utils.nerf_utils import write_nerf_data


def _plot_obs(obs) -> None:
    """ Plot observation for debug purposes """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for idx, im in enumerate(obs['color']):
        axs[idx].imshow(im)
    plt.show()


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']
    enable_nerf = cfg['enable_nerf']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        for step in range(task.max_steps):
            act = agent.act(obs, info)
            episode.append((obs, act, reward, info))

            if enable_nerf:
                # Defer writing NeRF data as it's expensive and
                # we only need it if the episode is successful.
                deferred.add(
                    write_nerf_data,
                    env=env,
                    dataset=dataset,
                    seed=seed,
                    step=step,
                    should_plot=False,
                )

            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)
            deferred.execute()


if __name__ == '__main__':
    # Logger setup
    debug = False
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level=logging.INFO)

    main()
