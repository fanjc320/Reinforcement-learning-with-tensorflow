"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import *
from RL_brain import QLearningTable
import datetime

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            begin = datetime.datetime.now()
            # fresh env
            env.render()
            end = datetime.datetime.now()
            usetime = (end - begin).microseconds
            # logger.error(usetime)

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            time.sleep(0.2)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    # file_handler = logging.FileHandler('logs.log')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # stdout_handler = logging.StreamHandler(sys.stdout)
    # stdout_handler.setLevel(logging.DEBUG)
    # stdout_handler.setFormatter(formatter)
    
    # logger.addHandler(stdout_handler)
    logger.info("q_learning_maze begin!")

    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()