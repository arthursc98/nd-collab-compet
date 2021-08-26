[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: https://pylessons.com/static/images/Reinforcement-learning/09_A2C-reinforcement-learning/Actor-Critic-fun.png "Actor Critic"
[image4]: https://cdn-media-1.freecodecamp.org/images/1*SvSFYWx5-u5zf38baqBgyQ.png "Advantage Function"
[image5]: https://paperswithcode.com/media/methods/b6cdb8f5-ea3a-4cca-9331-f951c984d63a_MBK7MUl.png "SARS Memory"
[image6]: https://miro.medium.com/max/700/1*vLFINWklJ0BtYtgzwK223g.png "Gradient Clipping"
[image7]: imgs/model_performance.png "Model Performance"
[image8]: imgs/model_epochs.png "Model Epochs"

# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Solution
Before we dive deep in my solution i'll cover the following agenda an try to cover all concepts used in the implementation.
- Actor-Critic
- Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
- Epsilon Greedy
- Experience Replay
- Ornstein-Uhlenbeck Process
- Gradient Clipping

## Actor-Critic
Have you ever heard about GAN's? Well they basically are the fake images that you see on the internet, one cool example is [this cat does not exist](https://thiscatdoesnotexist.com/) which shows up cat that are made by a GAN. GAN's extends for Generative Adversarial Network, by a given training set they try to generate something similar to their training set. This class of ML was developed by Ian Goodfellow and his colleagues in 2014 and throw years pass there some improvements in this field. The architecture of GAN's it's really interesting and it's replicated to our Actor-Critic model, in GAN's we have two models that work together, one it's the Generator and the other one it's the Discriminator, when the Generator creates new examples the Discriminator judge if it was a good representation based on the training set.<br>
In Actor-Critic we have a similar architecture, we have our Actor that will play around the enviroment and our Critic which will tell to our Actor if it was a good move to take in the enviroment, which it's pretty cool right?<br>
![Actor Critic][image3]<br>
Now we have a function called Advantage Function, which shows us how much better it is to take a specific action compared to the average, general action at the given state, and that what's the equation below tell us, if you wanna go deeper i highly recommend to check on some resources like [this](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b) one that goes deep in math concepts with some pseudocode<br>
![Advantage Function][image4]

## Multi-Agent Deep Deterministic Policy Gradient
DDPG is a RL technique that combines Q-Learning and Policy Gradients. While the actor is a policy network that takes the state as input and outputs the exact action instead of probability for each action to choose a strategy to pick one of them. The critic is a Q-value network that takes in state and action as input and outputs the Q-value. But for this case we have multiple agents to do it, which makes them compete and collaborate with each other. For this implementation i used a shared replay buffer to each agent see what others agents was going through.

## Epsilon Greedy
Before we talk about epsilon greedy we need to know a very popular dilemma called Exploration / Exploitation, this dilemma is one of the hardest to think about, let's say that each time we play we have to decide between explore even more our environment and see which series of actions would lead to a highest rewards or keep what we know about the environment and continue to do the action that belongs the highest rewards, now, what should we do? Start exploring our environment and more often begins to exploit it? It's there any possible way to estimate when we need to explore the environment? Or even a heuristic? Well you will see about Epsilon Greedy in the next section which tries to solve this problem.<br>
Now that you know more about exploration / exploitation dilemma we can explain how Epsilon Greedy works, let's say we have a probability for those two actions, what epsilon greedy tries to do it's to generate a randomness into the algorithm, which force the agent to try different actions and not get stuck at a local minimum, so to implemente epsilon greedy we set a epsilon value between 0 and 1 where 0 we never explore but always exploit the knowledge that we already have and 1 do the opposite, after we set a value for epsilon we generate a random value usually from a normal distribution and if that value is bigger than epsilon we will choose the current best action otherwise we will choose a random action to explore the environment.<br>
For this current project i used the following parameter to control the exploitation and generate a little noise to force the some exploration.<br>
```python
EPSILON = 1.0           # Epsilon Exploration / Exploit Parameter
EPSILON_DECAY = 1e-6    # Decay Epsilon while training
```

## Experience Replay
In order to try to solve rare events detection in our model, we store each experience from our agent in a memory and sample it randomly so our agent start to generalize better and recall rare occurrences. Also for better performance we could use mini-batch's to see how our model converge. The image below shows up how figuratively the memory would look.<br>
![SARS Memory][image5]
For the current problem i used a `Buffer Size of 100.000 or 1e5` if you have some Python experience

## Ornstein-Uhlenbeck Process
The Ornstein-Uhlenbeck process it's a stochastic process that envolves a modification random walk in continuous time, this process tends to converge to the mean function, which it's called mean-reverting, at first sight i never saw this before but it's really good since it's applied into continuous problems as like the one we are trying to solve<br>
In OU Process we have three highly important parameters which are they the $\mu$ that represents the mean where the noise will be generated will tend to drift towards, $\theta$ it's the speed that the noise will reach the mean and $\sigma$ it's volatility parameter, this came from physics concepts but it's really used in financial problems, one of current posts that i got based on it's [this one](https://www.maplesoft.com/support/help/maple/view.aspx?path=Finance%2FOrnsteinUhlenbeckProcess#bkmrk0).<br<>
I tried to replicate the parameters used in [this paper](https://arxiv.org/pdf/1509.02971.pdf) so i didn't tried too many variation from those parameters but should be an awesome action to take in the future, but for now i used the following parameters to our Ornstein-Uhlenbeck generator.<br>
```python
OU_MU = 0.0             # Ornstein-Uhlenbeck Mu Parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck Theta Parameter
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck Sigma Parameter
```

## Gradient Clipping
Gradient Clipping tries to solve one of biggest problems in Backpropagation that it's exploding/vanishing gradients which happens really often in RNN's for example, the idea behind it's pretty simple, if the gradient gets too large or too small, we can rescale to be between two values e.g. gradient represents a variable $g$ and clip represents a variable called $c$ should obey the following formula: $|g| >= c$<br>
![Gradient Clipping][image6]<br>
To implement it on our code it's pretty simple the following code it's from the implementation that i did on the project but this should give an idea.
```python
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
```

## Models Architeture
The Actor and Critic models where builded up with FC layers both with 256 with 3 layers but we have different inputs, for the actor we input a state and for the Critic we have both, the state and the action but let me show you how both models is currently working
```bash
Actor:
State -> FC 256 units ->  ReLU -> FC 256 units -> ReLU -> FC 2 units (Total Actions at this problem) -> TanH

Critic:
State -> FC 256 units -> ReLU -> Concatenate w/ Action -> FC 256 units -> ReLU -> FC 256 units
```
Also to optimize the problem you will see that i used two Learning Rates which for the Critic is 1e-3 and for the Actor is 1e-4 but while i was optimizing it i choose different learning rates to see if one of both networks would reachA the local maximum faster ou slower. Also to prevent overfitting into our models i used a Batch Size with 256 of size to evaluate and train our neural network with mini batches.<br>
Now for the last two parameters that i used the `TAU` and `GAMMA`, $\tau$ is used to soft update of target networks parameters and $\gamma$ is the discount factor for our rewards, so for a larger $\gamma$ we will care more about the distant rewards otherwise if we use a $\gamma$ like 0 we only care about the current reward, to solve this problem i used a $\tau$ with 1e-3(0.001) and $\gamma$ 0.99.<br>

## Model Performance
At first try i used the past hyperparameters from the last project but it wasn't a great success, which it's good right? It show us that everything was a unique best solution. But to comment a little bit here, i used all the past hyperparameters that i told right here and it gave me a result that was pretty similar from the benchmark implementation, it could be better but this can go to further experiments, right now i'm little bit busy with College but at my pov it was a really great result, that means it can get really better from now on. Let's detail it more at the next section<br>
![Model Epochs][image8]<br>
![Model Performance][image7]<br>

## Ideas for Future Work
Since a common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, for a more fast way is try to optimize for a longer time the hyperparameters or I would probably study about Twin Delayed DDPG or Soft Actor Critic and try to implement it since it's some concepts that are kinda new.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file and adapt the path in UnityEnviroment refering to the filename, just being generic to deal with OSX and other systems, me personally runned it on Udacity Enviroment but it's completely easy to do it on you're local machine. 

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
	
2. Since i already created a file with several dependencies that we need to run the project. First of all it's better to install the python folder which contains some stable libraries. To do so follow the next command lines.
```bash
cd python
pip install .
```

3. Now that the env already have some things that we need, let's install the other part of the dependencies
```bash
cd ../
pip install -r requirements.txt
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

6. Now that we have it all to run the project all you gotta do is to open the `Tennis.ipynb` and run each cell, it's highly recommended to run it with GPU for faster results.