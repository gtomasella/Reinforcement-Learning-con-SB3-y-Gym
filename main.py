# Gym stuff
import gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ta
from gym_anytrading.envs import StocksEnv
import os

import yfinance as yf
# Cargar los datos de ejemplo de acciones de Google (GOOGL)
df = yf.download("AAPL", start="2017-01-01", end="2023-04-30")
# df

#Ambiente de prueba random
window_size = 3
start_index = window_size
end_index = len(df)-300
env = gym.make('stocks-v0', df=df, frame_bound=(start_index, end_index), window_size=window_size)
#Ambiente de prueba random
state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print("info", info)
        break
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

print(env.action_space)
print(env.observation_space)
print(env.signal_features)
print(env.shape)
print(env.window_size)
print(env.reward_range)

# Calcular las medias móviles y el indicador estocástico utilizando la biblioteca TA
df['sma_10'] = ta.trend.sma_indicator(df['Close'], window=10)
df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
df=df.fillna(0)
df

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Open','High','Low','Close','Adj Close', 'Volume','sma_10', 'sma_20', 'sma_50','stoch']].to_numpy()[start:end]
    return prices, signal_features

df.loc[:, 'Close'].to_numpy()[start_index: end_index]

class MyCustomEnv(StocksEnv):
    _process_data = add_signals
    
window_size1 = 10
start_index = window_size1
end_index = len(df)-300   
env2 = MyCustomEnv(df=df,  frame_bound=(start_index, end_index), window_size=window_size1)

models_dir = "models/A2C"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env_maker = lambda: env2
env = DummyVecEnv([env_maker])
log_path = os.path.join('Training', 'Logs')

model = A2C('MlpPolicy', env, verbose=1,tensorboard_log=logdir) 
TIMESTEPS = 10000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

model.load(f"{models_dir}/{290000}")

start_index = len(df)-350   
end_index = len(df) 
env = MyCustomEnv(df=df, window_size=window_size1, frame_bound=(start_index,end_index))
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break
    
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()