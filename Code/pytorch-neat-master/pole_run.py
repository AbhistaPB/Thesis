import logging

import gym
import torch
import numpy as np
# from render_browser import render_browser

import neat.population as pop
import neat.experiments.pole_balancing.config as c
from neat.experiments.pole_balancing.Conv2explain import Converter
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
import pickle
from os import path


logger = logging.getLogger(__name__)

logger.info(c.PoleBalanceConfig.DEVICE)
neat = pop.Population(c.PoleBalanceConfig)
solution, generation = neat.run()
solution_path = './images/pole-balancing-solution.pkl'
solution_fitness_path = './images/pole-balancing-fitness.pkl'
if not path.exists(solution_path):
    with open(solution_path, 'wb') as f:
        pickle.dump(solution, f)
    with open(solution_fitness_path, 'wb') as f:
        pickle.dump(neat.best_fitness, f)
else:
    with open(solution_path.replace('.pkl', '-explained.pkl'), 'wb') as f:
        pickle.dump(solution, f)
    with open(solution_fitness_path, 'wb') as f:
        pickle.dump(neat.best_fitness, f)

# @render_browser
def Best_run(solution, logger):
    if solution is not None:
        logger.info('Found a Solution')
        draw_net(solution, view=True, filename='./images/pole-balancing-solution', show_disabled=True)

        # OpenAI Gym
        env = gym.make('CartPole-v1')
        done = False
        observation = env.reset()

        fitness = 0
        conv2 = Converter()
        phenotype = FeedForwardNet(solution, c.PoleBalanceConfig)
        explanation = []
        
        while not done:
            env.render()
            observation = np.array([observation])
            observation = conv2.obsconv(observation, c.PoleBalanceConfig.version)

            input = torch.Tensor(observation).to(c.PoleBalanceConfig.DEVICE)

            pred = round(float(phenotype(input)))
            observation, reward, done, info = env.step(pred)

            fitness += reward

            if c.PoleBalanceConfig.version != 'V0':
                explained = conv2.explain(c.PoleBalanceConfig.version)
                explanation.append(explained)
            else:
                explanation = [None]
            
            pred_human = 'left' if pred == 0 else 'right'

            logger.info('I can see ' + explained + '. Hence, I go ' + pred_human)

        env.close()
    return None

Best_run(solution, logger)