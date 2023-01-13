import numpy as np
import math

class Converter:
    def __init__(self) -> None:
            # Hard coded for the cartpole problem
            # Triangular membership function
            self.a = [[-2.4, -1.6, 0], [-5, -2.5, 0], [0.2095, -0.10475, 0], [-5, -2.5, 0]]
            self.b = [[-1.6, 0, 1.6], [-2.5, 0, 2.5], [-0.10475, 0, 0.10475], [-2.5, 0, 2.5]]
            self.c = [[0, 1.6, 2.4], [0, 2.5, 5], [0, 0.10475, 0.2095], [0, 2.5, 5]]

            # Gaussian membership function
            self.mean = [[-0.8, 0, 0.8], [-1.5, 0, 1.5], [-0.052375, 0, 0.052375], [-1.5, 0, 1.5]]
            self.sigma = [[0.5, 0.5, 0.5], [0.75, 0.75, 0.75], [0.075, 0.075, 0.075], [0.75, 0.75, 0.75]]
            self.fuzzy_obs = []

    def obsconv(self, observation, version, mem_func = 'Triangular'):
        if version == 'V0':
            observation = self.obsV0(observation)
        elif version == 'V1':
            observation = self.obsV1(observation)
        elif version == 'V2':
            observation = self.obsV2(observation)
        elif version == 'V3':
            observation = self.obsV3(observation, mem_func)
        return observation

    def obsV0(self, observation):
        return observation

    def obsV1(self, observation):
        self.pos_left = 1.0 if observation[0][0] < - 0.8 else 0.0
        self.pos_mid = 1.0 if ((observation[0][0] >= -0.8) and (observation[0][0] <= 0.8)) else 0.0
        self.pos_right = 1.0 if observation[0][0] > 0.8 else 0.0
        self.vel_high = 1.0 if ((observation[0][1] > 5) or (observation[0][1] < -5)) else 0.0
        self.vel_low = 1.0 if ((observation[0][1] <= 5) and (observation[0][1] >= -5)) else 0.0
        self.pole_left = 1.0 if observation[0][2] < -0.07 else 0.0
        self.pole_mid = 1.0 if ((observation[0][2] >= -0.07) and (observation[0][2] <=0.07)) else 0.0
        self.pole_right = 1.0 if observation[0][2] > 0.07 else 0.0
        self.avel_high = 1.0 if ((observation[0][3] < -5) or (observation[0][3] > 5)) else 0.0
        self.avel_low = 1.0 if ((observation[0][3] >= -5) and (observation[0][3] <= 5)) else 0.0
        new_observation = np.array([np.array([self.pos_left, self.pos_mid, self.pos_right, self.vel_high,\
            self.vel_low, self.pole_left, self.pole_mid, self.pole_right, self.avel_high, self.avel_low])])
        return new_observation
    
    def obsV2(self, observation):
        self.pos_left = observation[0][0] if observation[0][0] < - 0.3 else 0.0
        self.pos_mid = observation[0][0] if ((observation[0][0] >= -0.3) and (observation[0][0] <= 0.3)) else 0.0
        self.pos_right = observation[0][0] if observation[0][0] > 0.3 else 0.0
        self.vel_high = observation[0][1] if ((observation[0][1] > 2) or (observation[0][1] < -2)) else 0.0
        self.vel_low = observation[0][1] if ((observation[0][1] <= 2) and (observation[0][1] >= -2)) else 0.0
        self.pole_left = observation[0][2] if observation[0][2] < -0.03 else 0.0
        self.pole_mid = observation[0][2] if ((observation[0][2] >= -0.03) and (observation[0][2] <=0.03)) else 0.0
        self.pole_right = observation[0][2] if observation[0][2] > 0.03 else 0.0
        self.avel_high = observation[0][3] if ((observation[0][3] < -2) or (observation[0][3] > 2)) else 0.0
        self.avel_low = observation[0][3] if ((observation[0][3] >= -2) and (observation[0][3] <= 2)) else 0.0
        new_observation = np.array([np.array([self.pos_left, self.pos_mid, self.pos_right, self.vel_high,\
            self.vel_low, self.pole_left, self.pole_mid, self.pole_right, self.avel_high, self.avel_low])])
        return new_observation
    
    def obsV3(self, observation, mem_type):
        self.fuzzy_obs = []
        # fuzzy_obs = []
        [observation] = observation
        if mem_type.lower() == 'triangular':
            for i, x in enumerate(observation):
                for ind in range(3):
                    self.fuzzy_obs.append(self.triangular(x, self.a[i][ind], self.b[i][ind], self.c[i][ind]))

        elif mem_type.lower() == 'gaussian':
            for i, x in enumerate(observation):
                for ind in range(3):
                    self.fuzzy_obs.append(self.gaussian(x, self.mean[i][ind], self.sigma[i][ind]))
        
        # self.fuzzy_obs = fuzzy_obs
        return [self.fuzzy_obs]
    
    def triangular(self, x, a, b, c):
        y = max(min(((x-a)/(b-a)), ((c-x)/(c-b))), 0)
        return y

    def gaussian(self, x, mean, sigma):
        y = math.exp(-(0.5)*(((x - mean)/sigma)**2))
        return y
    def explain(self, version):
        explained = ''
        if (version == 'V1') or (version == 'V2'):
            explained += ' and pos_left' if self.pos_left != 0 else ''
            explained += ' and pos_mid' if self.pos_mid != 0 else ''
            explained += ' and pos_right' if self.pos_right != 0 else ''
            explained += ' and vel_high' if self.vel_high != 0 else ''
            explained += ' and vel_low' if self.vel_low != 0 else ''
            explained += ' and pole_left' if self.pole_left != 0 else ''
            explained += ' and pole_mid' if self.pole_mid != 0 else ''
            explained += ' and pole_right' if self.pole_right != 0 else ''
            explained += ' and avel_high' if self.avel_high != 0 else ''
            explained += ' and avel_low' if self.pole_left != 0 else ''
            return explained[5:]
        elif version == 'V3':
            explained += ' and left' if self.fuzzy_obs[0] != 0 else ''
            explained += ' and middle' if self.fuzzy_obs[1] != 0 else ''
            explained += ' and right' if self.fuzzy_obs[2] != 0 else ''
            explained += ' and vel_left' if self.fuzzy_obs[3] != 0 else ''
            explained += ' and vel_low' if self.fuzzy_obs[4] != 0 else ''
            explained += ' and vel_right' if self.fuzzy_obs[5] != 0 else ''
            explained += ' and pole_left' if self.fuzzy_obs[6] != 0 else ''
            explained += ' and pole_middle' if self.fuzzy_obs[7] != 0 else ''
            explained += ' and pole_right' if self.fuzzy_obs[8] != 0 else ''
            explained += ' and ang_vel_left' if self.fuzzy_obs[9] != 0 else ''
            explained += ' and ang_vel_low' if self.fuzzy_obs[10] != 0 else ''
            explained += ' and ang_vel_right' if self.fuzzy_obs[11] != 0 else ''
            return explained[4:]
        else:
            return None