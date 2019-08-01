import gym


# This agent keeps track of customer needs and dispatches trucks accordingly.
env = gym.make('gym_logistics_simple:logistics-simple-v0')

print(env.trucks)
print(env.customers)

# while not done:
