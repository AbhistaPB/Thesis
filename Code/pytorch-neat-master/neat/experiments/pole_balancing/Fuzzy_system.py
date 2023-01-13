from numpy import argmax, array
class Fuzzifier:
    def __init__(self) -> None:
        self.observations = []
        self.network_predictions = []
        self.decisions = []

    def track_observations(self, observation):
        self.observations.append(observation)
    
    def track_predictions(self, prediction):
        self.network_predictions.append(prediction)

    
    def decide(self, inference_right, inference_left):
        explainations = [' left', ' middle', ' right',\
                         ' vel_left', ' vel_low', ' vel_right',\
                         ' pole_left', ' pole_middle', ' pole_right',\
                         ' ang_vel_left', ' ang_vel_low', ' ang_vel_right']

        for observation in self.observations:
            val_right = 0
            for rule in inference_right:
                temp = 100000
                for i, literal in enumerate(explainations):
                    if literal in rule:
                        temp = min(observation[i], temp)
                val_right = max(temp, val_right)
            
            val_left = 0
            for rule in inference_left:
                temp = 100000
                for i, literal in enumerate(explainations):
                    if literal in rule:
                        temp = min(observation[i], temp)
                val_left = max(temp, val_left)
            
            self.decisions.append(round(argmax([val_left, val_right])))
    
        return self.decisions   

    def calculate_accuracy(self):
        return sum(array(self.decisions) == array(self.network_predictions))
