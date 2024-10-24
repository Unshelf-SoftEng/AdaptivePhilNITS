import os
import numpy as np
import pandas as pd
import random
from adaptive_learning import AdaptiveLearningEnv

# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

# Initialize environment with a mock question bank
question_bank = pd.DataFrame({
    'question_id': range(1, 11),
    'difficulty': np.random.normal(0, 1, 10),
    'discrimination': np.random.uniform(0.5, 2, 10)
})
env = AdaptiveLearningEnv(question_bank, learner_ability=0.5)


if os.path.exists('q_table.npy'):
    q_table = np.load('q_table.npy')
else:
    q_table = np.zeros((env.num_bins, env.action_space.n))


#training loop
for episode in range(num_episodes):
    state = env.reset()[0]  # Get initial state (discretized)
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit
        
        next_state, reward, done, _ = env.step(action)
        next_state = next_state[0]  # Extract the next state
        
        # Q-learning update rule
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        state = next_state
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
