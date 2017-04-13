import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):

    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0

    # The Q-Table learning algorithm
    while j < 99:
        j += 1

        # Choose an action by greedily (with noise) picking from Q table
        rand_shift = np.random.randn(1, env.action_space.n) * (1. / (i + 1))
        a = np.argmax(Q[s, :] + rand_shift)

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)

        # Update Q-Table with new knowledge
        # Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        Q[s, a] = r + y * np.max(Q[s1, :])

        rAll += r
        s = s1
        if d: break

    rList.append(rAll)

print("Score over time: {}".format(str(sum(rList) / num_episodes)))