import logging
from neat.experiments.pole_balancing.Fuzzy_system import Fuzzifier
import gym
import torch
import numpy as np

import neat.population as pop
import neat.experiments.pole_balancing.config as c
from neat.experiments.pole_balancing.Conv2explain import Converter
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
import numpy as np
from tqdm import tqdm
import neat.utils as utils

logger = logging.getLogger(__name__)
logger.info(c.PoleBalanceConfig.DEVICE)
neat = pop.Population(c.PoleBalanceConfig)

solution_path = './images/pole-balancing-solutionV3.pkl'
monte_runs = [25]

# def setup(neat, config):
#     if config:
#         # Get Fitness of Every Genome
#         for genome in neat.population:
#             fitness, explanation = neat.Config.fitness_fn(genome)
#             genome.fitness = max(0, fitness)
#             genome.explanation = explanation

#         best_genome = utils.get_best_genome(neat.population)
#         return best_genome
#     else:
#         return neat

def run(solution, logger, name):
    if solution is not None:
        if name == '__main__':
            logger.info('Found a Solution')
            draw_net(solution, view=True, filename='./images/pole-balancing-solution' + c.PoleBalanceConfig.version, show_disabled=True)

        # OpenAI Gym
        env = gym.make('CartPole-v1')
        done = False
        observation = env.reset()

        fitness = 0
        conv2 = Converter()
        fuzzifier = Fuzzifier()
        phenotype = FeedForwardNet(solution, c.PoleBalanceConfig)
        explanation_right = []
        explanation_left = []
        
        while not done:
            if name == '__main__':
                env.render()
            
            observation = np.array([observation])
            observation = conv2.obsconv(observation, c.PoleBalanceConfig.version)

            fuzzifier.track_observations(*observation)

            input = torch.Tensor(observation).to(c.PoleBalanceConfig.DEVICE)

            if c.PoleBalanceConfig.version == 'V3':
                out = torch.argmax(*phenotype(input))
            else:
                out = phenotype(input)

            pred = round(float(out))
            fuzzifier.track_predictions(pred)
            
            observation, reward, done, info = env.step(pred)

            fitness += reward

            if c.PoleBalanceConfig.version != 'V0':
                explained = conv2.explain(c.PoleBalanceConfig.version)
                if pred == 0:
                    explanation_left.append(explained)
                    pred_human = 'left'
                else:
                    explanation_right.append(explained)
                    pred_human = 'right'
                
                if name == '__main__':
                    logger.info('I can see ' + explained + '. Hence, I go ' + pred_human)
            else:
                explanation_right, explanation_left = [None], [None]

        env.close()
        if name == '__main__':
            logger.info('Train_accuracy: ' + str((solution.fitness)/10) + ' with train_fitness ' + str(solution.fitness))
            logger.info('Test_fitness ' + str(fitness))
        if c.PoleBalanceConfig.version == 'V3':
            fuzzy_output = fuzzifier.decide(explanation_right, explanation_left)
            correct_fuzzy_predictions = fuzzifier.calculate_accuracy()
            if name == '__main__':
                logger.info(fuzzy_output)
                logger.info(fuzzifier.network_predictions)
                logger.info('Fuzzy system: ' + str(correct_fuzzy_predictions) + ', Explanation length:' + str(len(solution.explanation)))
            return correct_fuzzy_predictions, solution.fitness
        return solution.reward, solution.fitness

if __name__ == '__main__':

    # with open(solution_path, 'rb') as f:
    #     solution_neat = pickle.load(f)

    # solution = setup(solution_neat, False)

    explanation_accuracy = []
    train_fitness = []
    for total_runs in monte_runs:
        explanation_run_accuracy = []
        train_run_fitness = []
        for i in tqdm(range(total_runs)):
            neat = pop.Population(c.PoleBalanceConfig)
            solution, generation = neat.run()
            explanation, fitness = run(solution, logger, '--')
            explanation_run_accuracy.append(explanation)
            if c.PoleBalanceConfig.version == 'V1':
                train_run_fitness.append(fitness - explanation)
            else:
                train_run_fitness.append(fitness)
        explanation_accuracy.append(sum(explanation_run_accuracy)/total_runs)
        train_fitness.append(sum(train_run_fitness)/total_runs)
    logger.info('Explanation accuracy = ' + str(explanation_accuracy[0]))
    logger.info('explanation_runs' + str(explanation_run_accuracy))
    logger.info('Fitness' + str(train_fitness))
    logger.info('All fitenesses' + str(train_run_fitness))
