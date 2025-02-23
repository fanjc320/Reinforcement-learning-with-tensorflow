"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from doctest import Example
import numpy as np
import pandas as pd
from maze_env import logger

class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        a=5.0
        b=5.0
        for i in range(3):
            for j in range(3):
                # if (a, b, a+30, b+30) in [(85.0, 85.0, 115.0, 115.0), (45.0, 85.0, 75.0, 115.0)]:
                if (a, b, a+30, b+30) in [(85.0, 85.0, 115.0, 115.0)]:
                    self.check_state_exist("terminal")
                else:
                    self.check_state_exist(str([a, b, a+30, b+30]))
                b += 40
                print("++++++++++ state:",a,b)
            a += 40
            b = 5.0
            
        print("self.q_table:", self.q_table)
        print("====================")
        

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:

            # choose best action # 和self.q_table.loc[observation]是不同的 
            state_action = self.q_table.loc[observation, :] # 一行的动作，即上下左右,0,1,2,3
            
            print(" typeof_state_action:", type(state_action))
            print(" state_action:",state_action)
# state_action: 0    0.000000
# 1    0.000000
# 2    0.004714
# 3    0.000000
# Name: [5.0, 125.0, 35.0, 155.0], dtype: float64
# np.max(state_action): 0.0047143164899028575
#  idxmax: 0    False
# 1    False
# 2     True
# 3    False
# Name: [5.0, 125.0, 35.0, 155.0], dtype: bool
#  ----idx: Int64Index([2], dtype='int64')  action: 2

# https://www.sharpsightlabs.com/blog/numpy-max/
# np.max：(a, axis=None, out=None, keepdims=False)
# 求序列的最值
# 最少接收一个参数
# axis：默认为列向（也即 axis=0），axis = 1 时为行方向的最值；

            print("np.max(state_action):", np.max(state_action))
            idxmax = (state_action == np.max(state_action))
            print(" idxmax:",idxmax)
#             Series.index¶
# The index (axis labels) of the Series.
#idx: Int64Index([1], dtype='int64')
            idx = state_action[idxmax].index
            # or idx = np.where(state_action == np.max(state_action))[0]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(idx) #这部分属于预测
            print(" ----idx:",idx," action:",action)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            print("------------------------------")
            print(" q_table:",self.q_table)
            print(" q_table _s:",s_)
            print(" q_table_loc:",self.q_table.loc[s_, :])
            s_max = self.q_table.loc[s_, :].max()
            print(" q_table_loc_max:",s_max)
            q_target = r + self.gamma * s_max  # next state is not terminal
            print(" q_target:", q_target, " r:", r)
        else:
            q_target = r  # next state is terminal
        #property DataFrame.loc
        #Access a group of rows and columns by label(s) or a boolean array.
        diff = q_target - q_predict
        oldtb = self.q_table.copy(deep = True)
        self.q_table.loc[s, a] += self.lr * diff  # update
        if not oldtb.equals(self.q_table):
            logger.info(self.q_table)

    def check_state_exist(self, state):

        if state not in self.q_table.index: # state like '[5.0, 5.0, 35.0, 35.0]'
            if state == [85.0, 85.0, 115.0, 115.0]: # fjc：和terminal重复了
                return
            #e.g:======check state: [5.0, 5.0, 35.0, 35.0]  q_table.columns: Int64Index([0, 1, 2, 3], dtype='int64')
            # print("======check state:",state, " q_table.columns:", self.q_table.columns)
            # append new state to q table
            self.q_table = self.q_table.append(
#                 The [0] * x creates a list with x elements. So,
# >>> [ 0 ] * 5
# [0, 0, 0, 0, 0]
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state, # if# TypeError: Can only append a Series if ignore_index=True or if the Series has a name
                )
            )
            print(" self.q_table:", self.q_table, " typeof qtable0:", type(self.q_table[0]))
            print(" qtable0:", self.q_table[0])
            print(" qtable0::", self.q_table[0:])
            print(" series name:", self.q_table[0].name) # 0 ????????

# pandas.Series.name¶
# property Series.name
# The name of a Series within a DataFrame is its column name.

# df = pd.DataFrame([[1, 2], [3, 4], [5, 6]],
#                   columns=["Odd Numbers", "Even Numbers"])
# df
#    Odd Numbers  Even Numbers
# 0            1             2
# 1            3             4
# 2            5             6
# df["Even Numbers"].name
# 'Even Numbers'

# name这个属性很奇怪，官网里是column的意思，实际使用dataframe.append的时候，也可以当做index来用,append 已经deprecated了
# here is another usage of the 'name' parameter. I will give an example. In this example we will see that the parameter 'name' could be used as an index name for values.

# purchase_1 = pd.Series({'Name': 'JJ',
#                         'Item': 'A',
#                         'Cost': 22.00})
# purchase_2 = pd.Series({'Name': 'KK',
#                         'Item': 'B',
#                         'Cost': 22.50})

# dfn = pd.DataFrame([purchase_1, purchase_2], index=['Store X', 'Store Y'])

# dfn = dfn.append(pd.Series(data={'Cost': 30.00, 'Item': 'C','Name': 'TT'}, name='Store Y'))
# dfn



# Out[3]: 
#          Cost Item Name
# Store X  22.0    A   JJ
# Store Y  22.5    B   KK
# Store Y  30.0    C   TT


# qtable的样式

