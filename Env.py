# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = list(permutations(range(1,m+1), 2)) 
        self.action_space.append([0,0])
        self.state_init = [1,0,0] 
        # Start the first round
        self.reset()
        
    def state_encod(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        pos = state[0]
        hour = state[1]
        day = state[2]
        posE = [0]*m
        hourE = [0]*t
        dayE = [0]*d
        posE[pos] = 1
        hourE[hour] = 1
        dayE[day] = 1
        state_encod = posE+hourE+dayE
        return state_encod


    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1: 
            requests = np.random.poisson(12)
        if location == 2: 
            requests = np.random.poisson(4)
        if location == 3: 
            requests = np.random.poisson(7)
        if location == 4: 
            requests = np.random.poisson(8)
        if requests >15:
            requests =15
        if requests <= 0:
            requests = 1
        possible_actions_idx = random.sample(range(1, (m-1)*m), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]
        return possible_actions_idx,actions   

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        action = self.action_space[action]
        startP = action[0]-1
        endP = action[1]-1
        currentP = state[0]
        currentH = state[1]
        currentD = state[2]
        
        if startP==0 and endP==0:
            return -1*C
   
        timeTo = Time_matrix[currentP][startP][currentH][currentD]
        
        currentHS = int((currentH + timeTo) % 24)
        currentDS = int((currentD+1)%7) if (currentH + timeTo)/24.0 >= 1.0 else int(currentD)
        
        timeFrom = Time_matrix[startP][endP][currentHS][currentDS]
        return R*timeFrom -1*C*(timeTo+timeFrom)


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        action = self.action_space[action]
        startP = action[0]-1
        endP = action[1]-1
        currentP = state[0]
        currentH = state[1]
        currentD = state[2]
        
        #update from current to start point
        time = Time_matrix[currentP][startP][currentH][currentD]
        currentH = int((currentH + time) % 24)
        currentD = int((currentD+1)%7) if (currentH + time)/24.0 >= 1.0 else int(currentD)
        
        #update from start point to end point
        time = Time_matrix[startP][endP][currentH][currentD]
        currentH = int((currentH + time) % 24)
        currentD = int((currentD+1)%7) if (currentH + time)/24.0 >= 1.0 else int(currentD)
        
        next_state = [endP,currentH,currentD]
        return next_state
    
    def step(self,state,action,Time_matrix):
        done = False
        new_state = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(state, action, Time_matrix)
        action = self.action_space[action]
        if action[0]==0 and action[1]==0:
            done = True
        return new_state, reward, done

    def reset(self):
        self.state_init = [1,0,0] 
