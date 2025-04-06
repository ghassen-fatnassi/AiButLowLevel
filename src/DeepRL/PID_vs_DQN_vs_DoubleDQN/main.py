import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
iterations=None
for _ in range(iterations):
   #action = policy(observation)
   action=None
   observation, reward, terminated, truncated, info = env.step(action)
   if terminated or truncated:
      observation, info = env.reset()
env.close()