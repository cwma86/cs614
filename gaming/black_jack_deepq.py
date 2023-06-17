#!/usr/bin/env python
import argparse
# import gym
import logging
import numpy as np
from black_jack import BlackJack
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
logging.basicConfig(
    format='%(asctime)s,%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


MODEL_PATH = "./black_jack_model"
# Define the Deep Q-Learning agent
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, model=None, verbose=True):
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.0001
        if model:
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = self._build_model()

        self.model_path = MODEL_PATH
        if model:
             self.model_path = model
        if verbose:
            self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Explore new options
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, memory):
        batch = np.random.choice(len(memory), self.batch_size, replace=False)

        # build training data
        batch_state = np.array([])
        batch_target = np.array([])
        for idx in batch:
            state, action, reward, done = memory[idx]
            target = reward
            target_f = self.model.predict(state, verbose=0)
            if not done:
                target = reward + self.gamma * np.amax(target_f[0])
            target_f[0][action] = target
            if not batch_state.size:
                batch_state=np.array(state)
                batch_target=np.array(target_f)
            else:
                batch_state = np.concatenate((batch_state, state), axis=0)
                batch_target = np.concatenate((batch_target, target_f), axis=0)

        # train model with mini batch
        self.model.fit(batch_state, batch_target, epochs=1, verbose=0)

        # Decay the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
def input():
    parser = argparse.ArgumentParser(
        prog='black_jack_learn',
        description='train a model to learn how to play blackjack'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=500,
        help="number of games to be played for training"
    )
    parser.add_argument(
        '--test',
        type=int,
        default=1000,
        help="number of games to be played for validation"
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="path to model"
    )
    parser.add_argument(
        '--cardState',
        action='store_true',
        help="If you want to include seen cards to the game state"
    )
    args = parser.parse_args()

    return args

def save_model(agent):
    black_jack_model = MODEL_PATH
    if args.model:
        black_jack_model = args.model
    logger.info(f"saving model to {black_jack_model}")
    agent.model.save(black_jack_model)

def main(args):
    # Initialize the environment
    # env = gym.make('Blackjack-v1', render_mode="none")
    env = BlackJack(cardState=args.cardState)
    state_size = 3  # Player's sum, dealer's showing card, and player has a usable ace
    if args.cardState:
        state_size = 13  # Player's sum, dealer's showing card, and player has a usable ace, 0-10 count of seen cards
        # 0-10 count
        # aces, twos, threes, fours, fives, sixes, sevens, eights, nines, tens
        # all cards can hold max 4 except tens which hold 16 (number of cards in a single deck)

    action_size = 2  # Stay (0) or Hit (1)
    batch_size = 128 # number of steps to be used for training
    agent = DQNAgent(state_size, action_size, batch_size, model=args.model)

    # Train the agent
    num_episodes = args.epoch
    memory = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        if episode % 10 == 0:
            logger.info(f"epoch {episode} game {state}")
            save_model(agent)
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _1, _2 = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            memory.append((state, action, reward, done))
            state = next_state
        if len(memory) > batch_size:
            agent.replay(memory)
            memory = []

    # Play the game with the trained agent
    win_count = 0
    number_of_games = args.test
    print(f"testing the trained model for {number_of_games} games")
    for i in range(number_of_games):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _1, _2 = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if reward >= 1.0:
                win_count = win_count + 1
            # env.render()

    logger.info(f"Win count {win_count} number of games{number_of_games}")
    logger.info(f"Win rate {(win_count/number_of_games)*100}%")

    save_model(agent)

if __name__ == '__main__':
    args = input()
    main(args)
