# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 0 ..... m-1
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        #number of total action space = 5*4 + 1 =21 ; number of city =5
        # action_space = (city1,city2)
        self.action_space = [(c1,c2) for c1 in range(m) for c2 in range(m) if c1!=c2 or c1==0]
        
        # number of state space = 5*24*7 = 840 
        # state_space = (city, hour of the day , day of the week)
        self.state_space = [(c,h,day) for c in range(m) for h in range(t) for day in range(d)]
        
       
        # state_init = random state 
        #self.state_init = random.choice(self.state_space)
        # this makes the tracking easier
        self.state_init = random.choice([(0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)])
        
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_encod = [0] * (m+t+d)  # Initialize one hot encoded state
        state_encod[state[0]] = 1    # city
        state_encod[m+ state[1]] = 1  # time of the day 24hour
        state_encod[m+ t +state[2]] = 1  # day of the week 

        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        req =0
        location = state[0]
        if location == 0:
            req = np.random.poisson(2)
        if location == 1:
            req = np.random.poisson(12)
        if location == 2:
            req = np.random.poisson(4)
        if location == 3:
            req = np.random.poisson(7)
        if location == 4:
            req = np.random.poisson(8)
            
        # upper limit for requests = 15
        if req >15:
            req =15

        # (0,0) is not considered as customer request
        possible_actions_idx = random.sample(range(1, (m-1)*m +1), req) # random sample indices of actions
        actions = [self.action_space[i] for i in possible_actions_idx]
        
        # append offline state
        if (0, 0) not in actions:
            actions.append((0,0))
            possible_actions_idx.append(0) # append the index of (0,0)
        
        return possible_actions_idx,actions   

    def time_day(self, time, day, duration):
        """
        To update time and day after getting the time from time matrix
        """
        # bring time in 0-23 range
        time_updated =( time +int(duration)) % t    
        # when the journey happens for more than a day
        change_day = (time +int(duration)) // t 
        # bring the day in 0-6 range
        day_updated = (day + change_day ) % d
                   
        return time_updated, day_updated

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        """   
        reward = -Cost of battery used , when no ride happens(off line)
        OR
        reward = (revenue earned from pickup point 𝑝 to drop point 𝑞) - 
                 (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝)-
                 (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞)    """
        
        
        curr_loc = state[0]   # current loction, time and day from state
        curr_time = state[1]
        curr_day = state[2]
        pick_loc = action[0]  #pick up location
        drop_loc = action[1]  #drop location
        pick_time =0      # time of pick-up
        pick_day=0        # day when the pick-up happens
        
        # find time taken to reach pick up point from current location
        if curr_loc == pick_loc:
            t1=0  # time to reach pick-up point from current location
        else:
             # time to reach pick-up point from current location
            t1 = int(Time_matrix[curr_loc][pick_loc][curr_time][curr_day]) 
        # update time and day to reach the pick-up point 
        pick_time ,pick_day = self.time_day(curr_time , curr_day , t1)
        # time to reach drop point from pick-up point  
        t2 = int(Time_matrix[pick_loc][drop_loc][pick_time][pick_day])   
                 
        if ((pick_loc == 0) and (drop_loc ==0)):  # offline
            reward = -C    # -ve reward for offline
        else:
            reward = (R * t2) -( C * (t1 + t2))
    
        return reward

   
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        time_to_pick = 0   # time to reach pick-up point
        time_to_drop = 0   # time to reach drop point from pick-up point
        trans_time =0      # change in time after pick-up
        trans_day = 0      # change in day after pick-up
        time_taken =0      # total time taken 
        
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pick_loc = action[0]
        drop_loc = action[1]
        
        # case 1 :when in off line (0,0) state
        if (pick_loc == 0) and (drop_loc ==0):
            time_to_drop = 1  # here wait time is 1
            # update time and day 
            next_time ,next_day = self.time_day(curr_time , curr_day , time_to_drop)
            next_loc = curr_loc
                 
        else:   
            # case 2 : when the cab is already at the pick-up location 
            if curr_loc == pick_loc:
                time_to_pick =0           # time to reach pick-up point 
                trans_time = curr_time    # no change in time and day when the driver is already present at pick-up point 
                trans_day = curr_day 
             
            # case 3 : when cab needs to go to pick-up point  
            else:
                time_to_pick = int(Time_matrix[curr_loc][pick_loc][curr_time][curr_day])  # time to reach pick-up point 
                # update time and day after reaching the pick-up point
                trans_time ,trans_day = self.time_day(curr_time , curr_day , time_to_pick)
                    
            # time to reach the drop point    
            time_to_drop = int(Time_matrix[pick_loc][drop_loc][trans_time][trans_day])
            # update time and day after reaching the drop point
            next_time ,next_day = self.time_day(trans_time , trans_day , time_to_drop)
            next_loc = drop_loc
            
        time_taken = time_to_pick + time_to_drop
        next_state = (next_loc, next_time, next_day)
        return  next_state , time_taken
        
          
    def reset(self):
        return self.action_space, self.state_space, self.state_init
