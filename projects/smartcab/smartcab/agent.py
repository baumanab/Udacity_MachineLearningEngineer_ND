from __future__ import print_function
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import q_learn as qlearn
from pprint import pprint
import pickle






class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Initialize any additional variables
        self.ai= None # reset
        self.actions= self.env.valid_actions
        self.ai = qlearn.QLearn(self.actions, alpha=0.1, gamma=0.9, epsilon=0.1) 
        # create a container to hold metric data
        self.data_dict= dict(successes= list(), lenq= list(), qtables= list())      
 

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
    def data_collector(self, location, destination, qtable, data_dict):
        
        '''Accepts location, destination, qtable, and a data container (dict of
        lists) as inputs. The function appends the container lists with important
        metric information which will bee used later for performance analysis.
        '''
        
        data_dict['qtables'].append(qtable)
        data_dict['lenq'].append(len(qtable))
        
        if location == destination:
            data_dict['successes'].append(1)
        else:
            data_dict['successes'].append(0)
    
    
    def dump_pkl(self, my_object, filename):
        with open(filename, "w") as outfile:
            pickle.dump(my_object, outfile)
  

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update current state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        # Select action according to policy
        #valid_actions = [None, 'forward', 'left', 'right']
        #action = random.choice(valid_actions)
        action = self.ai.chooseAction(self.state)
             

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # Define the new state
        next_waypoint= self.planner.next_waypoint()
        next_inputs = self.env.sense(self)
        next_state = (next_inputs['light'], next_inputs['oncoming'], inputs['left'], inputs['right'], next_waypoint)

        # Learn policy based on state, action, reward
        self.ai.learn(self.state, action, reward, next_state)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        # aceess variables for metrics
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]  
        
        self.data_collector(location, destination, self.ai.q, self.data_dict)
        
                
        
        
            

def run():

    """Run the agent for a finite number of trials."""
    
      

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline= True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    a.dump_pkl(a.data_dict, 'data_dict.pkl')
    
    
    
 
    


if __name__ == '__main__':
    run()
