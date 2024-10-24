import gym
from gym import spaces
import numpy as np

class AdaptiveLearningEnv(gym.Env):
    def __init__(self, question_bank, learner_ability, num_bins=10):
        super(AdaptiveLearningEnv, self).__init__()
        self.question_bank = question_bank
        self.num_questions = len(question_bank)
        self.learner_ability = learner_ability
        self.current_ability = learner_ability
        self.num_bins = num_bins
        
        # Discretize the learner's ability space into bins
        self.ability_min = -3  # Minimum ability level
        self.ability_max = 3   # Maximum ability level
        self.action_space = spaces.Discrete(self.num_questions)
        self.observation_space = spaces.Box(low=self.ability_min, high=self.ability_max, shape=(1,), dtype=np.float32)

    def _discretize_state(self, ability):
        # Convert the continuous ability to a discrete bin index
        ability_scaled = (ability - self.ability_min) / (self.ability_max - self.ability_min)
        discrete_state = int(np.floor(ability_scaled * self.num_bins))  # Scale to num_bins
        return min(max(discrete_state, 0), self.num_bins - 1)  # Ensure it's within bounds
    
    def reset(self):
        self.current_ability = self.learner_ability
        return np.array([self._discretize_state(self.current_ability)])
    
    def step(self, action):
        question = self.question_bank.iloc[action]
        probability_correct = self._probability_correct(self.current_ability, question)
        correct = np.random.rand() < probability_correct
        reward = 1 if correct else -1
        self.current_ability += reward * 0.1
        done = False
        return np.array([self._discretize_state(self.current_ability)]), reward, done, {}
    
    def _probability_correct(self, ability, question):
        difficulty = question['difficulty']
        discrimination = question['discrimination']
        return 1 / (1 + np.exp(-discrimination * (ability - difficulty)))

