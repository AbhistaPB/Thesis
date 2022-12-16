import torch
import gym
import numpy as np
import pickle
from os import path

from neat.phenotype.feed_forward import FeedForwardNet


class PoleBalanceConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    log_wandb = True
    version = 'V2'
    name = 'Pure NEAT'
    solution_path = './images/pole-balancing-solution.pkl'
    Environment = 'CartPole-v1'
    
    if not path.exists(solution_path):
        FITNESS_THRESHOLD = 500.0
        NUM_INPUTS = 4
    else:
        FITNESS_THRESHOLD = 1000.0
        if version == 'V1':
            NUM_INPUTS = 10
        elif version == 'V2':
            NUM_INPUTS = 10
        else:
            NUM_INPUTS = 4
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 10
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    # Allow episode lengths of > than 200
    gym.envs.register(
        id='LongCartPole-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100000
    )

    def fitness_fn(self, genome):
        # OpenAI Gym
        env = gym.make(self.Environment)
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNet(genome, self)
        if path.exists(self.solution_path):
            with open(self.solution_path, 'rb') as f:
                solution = pickle.load(f)
            solution_phenotype = FeedForwardNet(solution, self)
            explanation_left = []
            explanation_right = []

        while not done:
            observation = np.array([observation])
            if path.exists(self.solution_path):
                if self.version == 'V1':
                    pos_left = 1.0 if observation[0][0] < - 0.8 else 0.0
                    pos_mid = 1.0 if ((observation[0][0] >= -0.8) and (observation[0][0] <= 0.8)) else 0.0
                    pos_right = 1.0 if observation[0][0] > 0.8 else 0.0
                    vel_high = 1.0 if ((observation[0][1] > 5) or (observation[0][1] < -5)) else 0.0
                    vel_low = 1.0 if ((observation[0][1] <= 5) and (observation[0][1] >= -5)) else 0.0
                    pole_left = 1.0 if observation[0][2] < -0.07 else 0.0
                    pole_mid = 1.0 if ((observation[0][2] >= -0.07) and (observation[0][2] <=0.07)) else 0.0
                    pole_right = 1.0 if observation[0][2] > 0.07 else 0.0
                    avel_high = 1.0 if ((observation[0][3] < -5) or (observation[0][3] > 5)) else 0.0
                    avel_low = 1.0 if ((observation[0][3] >= -5) and (observation[0][3] <= 5)) else 0.0
                    new_observation = np.array([np.array([pos_left, pos_mid, pos_right, vel_high, vel_low,\
                                                pole_left, pole_mid, pole_right, avel_high, avel_low])])
                elif self.version == 'V2':
                    pos_left = observation[0][0] if observation[0][0] < - 0.8 else 0.0
                    pos_mid = observation[0][0] if ((observation[0][0] >= -0.8) and (observation[0][0] <= 0.8)) else 0.0
                    pos_right = observation[0][0] if observation[0][0] > 0.8 else 0.0
                    vel_high = observation[0][1] if ((observation[0][1] > 5) or (observation[0][1] < -5)) else 0.0
                    vel_low = observation[0][1] if ((observation[0][1] <= 5) and (observation[0][1] >= -5)) else 0.0
                    pole_left = observation[0][2] if observation[0][2] < -0.07 else 0.0
                    pole_mid = observation[0][2] if ((observation[0][2] >= -0.07) and (observation[0][2] <=0.07)) else 0.0
                    pole_right = observation[0][2] if observation[0][2] > 0.07 else 0.0
                    avel_high = observation[0][3] if ((observation[0][3] < -5) or (observation[0][3] > 5)) else 0.0
                    avel_low = observation[0][3] if ((observation[0][3] >= -5) and (observation[0][3] <= 5)) else 0.0
                    new_observation = np.array([np.array([pos_left, pos_mid, pos_right, vel_high, vel_low,\
                                                pole_left, pole_mid, pole_right, avel_high, avel_low])])
            else:
                new_observation = observation
            

            input_new = torch.Tensor(new_observation).to(self.DEVICE)
            input = torch.Tensor(observation).to(self.DEVICE)
            
            pred = round(float(phenotype(input_new)))
            if path.exists(self.solution_path):
                sol_pred = round(float(solution_phenotype(input)))
                reward_extra = sol_pred*pred + (1-sol_pred)*(1-pred)
            else:
                reward_extra = 0

            observation, reward, done, info = env.step(pred)

            fitness += reward + reward_extra
            if self.version != 'V0':
                explained = ''
                explained += ' and pos_left' if pos_left != 0 else ''
                explained += ' and pos_mid' if pos_mid != 0 else ''
                explained += ' and pos_right' if pos_right != 0 else ''
                explained += ' and vel_high' if vel_high != 0 else ''
                explained += ' and vel_low' if vel_low != 0 else ''
                explained += ' and pole_left' if pole_left != 0 else ''
                explained += ' and pole_mid' if pole_mid != 0 else ''
                explained += ' and pole_right' if pole_right != 0 else ''
                explained += ' and avel_high' if avel_high != 0 else ''
                explained += ' and avel_low' if pole_left != 0 else ''
                if sol_pred == 1.0:
                    explanation_right.append(explained[5:])
                elif sol_pred == 0.0:
                    explanation_left.append(explained[5:])


        env.close()

        return fitness, explanation_right
