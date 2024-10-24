import os
import numpy as np
import pandas as pd
import random
from adaptive_learning import AdaptiveLearningEnv

# Define the categories
categories = [
    "Basic Theory", "Computer System", "Technology Element",
    "Development Technology", "Project Management", "Service Management",
    "Business Strategy", "System Strategy", "Corporate and Legal Affairs"
]

# Create a question bank with 80 questions
question_bank = pd.DataFrame({
    'question_id': range(1, 81),  # 80 questions
    'difficulty': np.random.normal(0, 1, 80),  # Random difficulty values
    'discrimination': np.random.uniform(0.5, 2, 80),  # Random discrimination values
    'category': np.random.choice(categories, 80)  # Random categories assigned to questions
})

# Function to estimate learner ability based on mock exam results, considering question difficulty
def estimate_learner_ability(mock_exam_scores, question_bank):
    correct_answers = 0
    total_difficulty = 0
    category_performance = {category: 0 for category in categories}  # Initialize performance per category

    # Update category performance based on the mock exam scores
    for question_id, score in mock_exam_scores:
        question = question_bank[question_bank['question_id'] == question_id]
        if not question.empty:
            category = question['category'].values[0]
            category_performance[category] += score
            if score == 1:
                correct_answers += 1
                # Get the difficulty of the question to weight the score
                difficulty = question['difficulty'].values[0]
                total_difficulty += difficulty

    # Calculate estimated ability as a weighted sum of correct answers over total difficulty
    estimated_ability = total_difficulty / len(mock_exam_scores) if len(mock_exam_scores) > 0 else 0
    print(f"Mock exam completed. Correct answers: {correct_answers}, Estimated ability: {estimated_ability:.2f}")
    
    return estimated_ability, category_performance

# Sample mock exam scores (question_id, 1 for correct, 0 for incorrect)
mock_exam_scores = [
    (1, 1), (2, 0), (3, 1), (4, 1), (5, 0), (6, 1), (7, 0), (8, 1),
    (9, 1), (10, 0), (11, 1), (12, 0), (13, 1), (14, 1), (15, 0), (16, 1),
    (17, 0), (18, 1), (19, 1), (20, 0), (21, 1), (22, 1), (23, 0), (24, 1),
    (25, 1), (26, 0), (27, 1), (28, 0), (29, 1), (30, 1), (31, 0), (32, 1),
    (33, 1), (34, 0), (35, 1), (36, 1), (37, 0), (38, 1), (39, 0), (40, 1),
    (41, 1), (42, 0), (43, 1), (44, 0), (45, 1), (46, 1), (47, 0), (48, 1),
    (49, 1), (50, 0), (51, 1), (52, 1), (53, 0), (54, 1), (55, 1), (56, 0),
    (57, 1), (58, 0), (59, 1), (60, 1), (61, 0), (62, 1), (63, 1), (64, 0),
    (65, 1), (66, 0), (67, 1), (68, 1), (69, 0), (70, 1), (71, 0), (72, 1),
    (73, 1), (74, 0), (75, 1), (76, 1), (77, 0), (78, 1), (79, 0), (80, 1),
]

# Estimate the learner's ability based on the mock exam scores
learner_ability, category_performance = estimate_learner_ability(mock_exam_scores, question_bank)

# Initialize the adaptive learning environment
env = AdaptiveLearningEnv(question_bank, learner_ability)

# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000


if os.path.exists('q_table.npy'):
    print("Loading existing Q-table...")
    q_table = np.load('q_table.npy')
else:
    print("Initializing new Q-table...")
    q_table = np.zeros((1, env.action_space.n))


# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:

        state_index = 0

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state, :])  # Exploit

        next_state, reward, done, _ = env.step(action)

         # Update Q-value; convert next_state to integer index as needed
        next_state_index = 0  # Modify this based on how you represent next state
        q_table[state_index, action] += alpha * (reward + gamma * np.max(q_table[next_state_index, :]) - q_table[state_index, action])

        state = next_state

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Epsilon: {epsilon:.4f}")

    # Save Q-table periodically every 100 episodes
    if (episode + 1) % 100 == 0:
        np.save('q_table.npy', q_table)
        print(f"Q-table saved after Episode {episode + 1}")


# Save Q-table at the end of training
np.save('q_table.npy', q_table)
print("Training completed, Q-table saved.")