import random

class QLearn:

    '''
    Code and approach modified from 

    https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/

    and 

    https://github.com/studywolf/blog/blob/master/RL/Cat%20vs%20Mouse%20exploration/qlearn_mod_random.py


    '''

    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
    
        self.q = dict() #initialze q lookup table as python dict

        self.epsilon = epsilon # wildcard
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount future reward
        self.actions = actions

    def getQ(self, state, action):
        
        # just a dict lookup
        return self.q.get((state, action), 0.0)

    def updateQ(self, state, action, reward, value):
    
        priorQ = self.q.get((state, action), None)
        if priorQ is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = priorQ + self.alpha * (value - priorQ)

    def chooseAction(self, state):
    
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)
        
        # compare random 0 - 1 to epsilon
        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]

        return action

    def learn(self, state, action, reward, next_state):
    
        maxqnew = max([self.getQ(next_state, a) for a in self.actions])
        
        # calculate value with Bellman equation
        value= reward + self.gamma * maxqnew
        
        # pass to Q update function
        self.updateQ(state, action, reward, value)