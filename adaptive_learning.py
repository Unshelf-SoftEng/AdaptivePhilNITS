import gym
from gym import spaces
import numpy as np

class AdaptiveLearningEnv(gym.Env):
    def __init__(self, question_bank, learner_ability):
        super(AdaptiveLearningEnv, self).__init__()
        self.question_bank = question_bank
        self.num_questions = len(question_bank)
        self.learner_ability = learner_ability
        self.current_ability = learner_ability
        self.action_space = spaces.Discrete(self.num_questions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
    def reset(self):
        self.current_ability = self.learner_ability
        return np.array([self.current_ability])
    
    def step(self, action):
        question = self.question_bank.iloc[action]
        probability_correct = self._probability_correct(self.current_ability, question)
        correct = np.random.rand() < probability_correct
        reward = 1 if correct else -1
        self.current_ability += reward * 0.1  # Adjust ability based on reward
        done = False
        return np.array([self.current_ability]), reward, done, {}
    
    def _probability_correct(self, ability, question):
        difficulty = question['difficulty']
        discrimination = question['discrimination']
        return 1 / (1 + np.exp(-discrimination * (ability - difficulty)))
