# Training a Smart Cab to Drive

## Preface

The object of this project is to train a smartcab to drive.  Notice this is train,
 not teach or tell the smartcab to drive, but train it.We want our agent, a smartcab 
 to get from point A to point B in duration (elpased time) x. The smartcab is 
 operating in a gridworld of streets, traffic lights, and other agents, where 
 the other agents are other cars.  The perspective of the cab is egocentric. 
 That is, it can only observe what is going on in its immediate environment. So,
 how do we get the cab to go from point A to point B?  There are three main options:

1. Tell: We could write a set of rules using flow control to force the cab to 
make certain decisions in every case. For example, this is what you do at a red
 light with oppsosing traffic, or without opposing traffic, etc. etc.  The challenge 
 with this approach is that it doesn't take very much to create an environment 
 that is so complex it would be daunting to think of every possiblilty and train 
 it. We would also have to hard code in new elements encountered into the environment.

2. Teach: Teaching woudl be akin to a supervised learning approach.  We could 
have agents drive around the virtual grid world and experince things and then use 
the record of those experinces to teach a new agent what to do.  This would be 
daunting for even fairly simple real-world applications and would not necesarily 
account for all new environmental elements

3. Train: Training is just what it sounds like. The agent drives around and experinces 
things. Some actions turn out great, some not so great. The agent stores these 
experiences and uses them to form a policy of behavior. That is, given a situation, 
what is the best action to take?


## Approach and Resources

So how do we do this? Through reinformcement learning.  Specifically we use Markov 
decision proceses and unsupervised learning to train our agent.  The approach I
have chosen for this task is Q learning.  There are many resources that explain 
Q learning, but these are the three I found most useful for building intuition as 
well as implementing Q learning in code.

1. The [Study Wolf Blog](https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/) 
blog contains a great overview of reinforcement learning and Q-learning and even 
a [python code implementation](https://github.com/studywolf/blog/tree/master/RL/Cat%20vs%20Mouse%20exploration)

2. [This](http://mnemstudio.org/path-finding-q-learning-tutorial.htm) great tutorial 
shows how Q-learning tables could be applied to teach an agent to move from a 
randomly selected room, out of a house.

3. [Demystifying Deep Reinforcement Learning](http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/) 
is another great resource with explanations of how Q learning, reinforcement learning, 
and Markov decision proceses relate. 


The resources listed above do a comprehensive job of explaining the principles 
and approach to reinforcement and Q learning but I paraphrase below, per my understanding. 
Essentially you have an agent who is operating on a grid or network, where locations 
are nodes and pathes are edges. In the case of a smartcab each intersection would 
be a node and each road an edge. The primary agent has a goal, in our case to 
go from point A to the destination. In this gridworld there are states. The states 
in our case are made up of the following elements [traffic signal, position of other 
agents relative to our agent]. Where traffic signal has the states [red, green] and a 
car can be to our left, right, or in front us. Our actions can be to stay put,
turn left, turn right, or go forward. From the gridworld perspective our agent
can choose to stay put or to advance in one of the four cardinal directions.

Each state, action combination (s,a) is associatted with a reward. This reward is 
an integer where the sign denotes a good vs bad action and the magnitude represents
how good or bad the action is given the state. So, our car is like pac man, running around
it's grid world and gobbling up  rewards, trying not to run into any ghosts. When the 
agent reaches the destination (talk about rewards and penalties etc. etc. and thatall other grid points couldbe zer
zeo)
