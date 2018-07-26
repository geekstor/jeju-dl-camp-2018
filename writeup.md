## jeju-dl-camp-2018

Authors: Valliappa Chockalingam and Rishab Gargeya

### Code Overview

* Trainer
  * train.py: Execute this file to run the entire project.
    * Flags:
    * Configuration Script: Location of configuration file.
  * Manager Class:
    * Self Variables: Env, Agent.
    * Run Method: Train multiple episodes. Visualize distributions (saves graph) with distagent.


* Configuration
  * All variabes for configuration: Network, Environment, Agent, Optimizer, Experience Replay.
  * Configuration file is propagated through trainer.py into each submodule folder (Agent, Optimizer, Memory, Environment).

* Agent
  * Agent creates the Network, Optimizer, Memory and explores the Environment.
  * Types: Agent, Categorical Agent, Quantile Regression, VAE
  * Methods: Act, Greedly Learning, Distribution, Learn

* Action Policy
  * Implements Policies
  * Methods: Policy, Epsilon Greedy

* Optimizer
  * Implements Optimizers for Network
  * Methods: Adam
  * Features: Gradient Clipping

* Memory
  * Implements Experience Replay
  * Methods: Add, Sample Size

* Environment
  * Wrapper for OpenAI Gym
  * Methods: Step, Reset, Render, Num_Actions, Observation_Dims

* Util
  * Util Functions
  * Methods: TF Copy OP

* GCP Config
  * Config files for running experiments on GCP
  * setup.py, config.yml, submit.sh : To run GCP CMLE ./submit.sh

* Function Approximator
  * Networks to learn relationships. Agent creates the network,
  * Head: Action Layer for Policy or Value.
  * Network: Algorithmic Function Approximator. Generic MLP / CNN implementation.
  * Types: Categorical DQN, Quantile Regression DQN, Implicit Quantile Networks (IQN)

* environment.yml : dependencies list

* run on GCP: nohup xvfb-run -s "-screen 0 1400x900x24" python -m trainer.train params_*.json &

This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/). We also thank our mentors Yu-Han Liu and Taehoon Kim.


### Camp Write-Up



Evaluating Safety and Generalization with Distributional RL Agents
Valliappa Chockalingam and Rishab Gargeya
Mentors: Yu-Han Liu and Taehoon Kim

About Us
Valliappa Chockalingam
Background in Deep Reinforcement Learning and Complex Systems. Interested in improving the stability, interpretability, generalization and trustworthiness of RL agents. 
Rishab Gargeya
Background in computer vision applied to large medical imaging datasets. Currently interested in scalable RL research and applications in meta-learning and safety.

Motivation
Consider the following two slot machines: 

Slot Machine A: At Least 100$
Slot Machine B: At Least 1000$
Question: Which would you pick?
Easy! If the goal is to maximize the amount of money one can expect (the expected utility), we just select Slot Machine B as it will always lead to a better outcome under the given specification.

Motivation
Now, consider the choice between these two slot machines: 

Slot Machine A: At Most 100$
Slot Machine B: At Most 1000$
Question: Which would you pick?
Well... What is the average case, best and worst case scenarios from pulling each of the arms? 

How do the distributions look?
Money (Reward / Return)
Probability

Theory - Deep Q-Networks



Expectations...
Mean values are sensitive to outliers!

Mean itself maybe unattainable as a return!

Why not consider the full distribution?

Problem
Small Loss Value ⇎ Better performance?
We can’t explicitly make inferences about or understand agents’ confidence in their decision making processes.
Image from: https://stats.stackexchange.com/questions/313876/loss-not-decreasing-but-performance-is-improving
Reinforcement learning is the study of teaching machines to take actions on their own. But we can’t explicitly control them.

Theory - Distributional RL

Main Idea: Learn with full distributions as opposed to expectations.
Right Top Corner: 
Quantile Regression 
DQN has a fixed
number of atoms.
Left: Categorical DQN has a fixed number of atoms and fixed support.
Right: Implicit Quantile Networks model the full distribution using 𝜏 ~ U[0, 1] and use a function ꞵ(𝜏) for modeling risk sensitivity. 
Credits: Left Top image: A Distributional Perspective on Reinforcement Learning (Bellemare et. al), Right Top Corner: distributional-dqn (Silvicek - github)

Sampling Distributions
Identity (Risk Neutral):  
ꞵ(𝜏) = 𝜏

Cumulative Prob. Weighting (CPW):  
ꞵ(𝜏; 𝜂) = 𝜏𝜂 / (𝜏𝜂 + (1 - 𝜏𝜂))1/𝜂

Wang (Risk Averse for 𝜂 < 0):  
ꞵ(𝜏; 𝜂) = 𝚽(𝚽-1(𝜏) + 𝜂)

Power (Risk Averse for 𝜂 < 0):  
ꞵ(𝜏) = 𝜏a for  𝜂 ≥ 0, 1 - (1 - 𝜏)a otherwise
where a = 1 / (1 + |𝜂|)

CVaR (Risk Averse):  ꞵ(𝜏; 𝜂) = 𝜂𝜏


Incorporating Sampling Distributions

Our Work and Timeline
06/01 -> 06/15
Literature Survey on Distributional RL
Full Implementations of seminal distributional reinforcement learning algorithm (Categorical DQN)
06/15 -> 07/01
Full Implementations most recent distributional RL agents (Quantile Regression, Implicit Quantile Networks)
Investigating Risk Adaptiveness through various sampling distribution approaches
07/01 -> 07/15
Implementation of Risk Adaptiveness with LSTM
Testing implementations with toy cartpole example
07/15 -> 07/27
Implementation of new loss function with Moment Matching Networks
Implementation of Uncertainty-Based Exploration (vs. epsilon-greedy exploration)
Implementation of Risk-Adaptive IQN on Maze Gridworld
Computation of Phase I Results (Cartpole, Maze)

Learning the sampling distribution parameters
Learning 𝜂 in the sampling distributions defined before using the current state as information.
Learning the quantiles in an active learning manner where a recurrent network chooses each quantile.
Shaded blocks denotes trainable variables

Risk Adaptive Implicit Quantile Networks
State. Agent about to lose.
Next State. Agent continues performing worse. More uncertainty in the value.
a
b
a
b
a
b
a
b
Let’s learn the sampling distribution!
Adaptive Risk Sensitivity with varying distorted expectations: ꞵ(𝜏) = a + bᐧ𝜏


Loss Functions
Categorical DQN made use of Softmax Cross Entropy.
Quantile Regression DQN and Implicit Quantile Networks made use of an asymmetric huber loss.

In some sense, if we are trying to match two distributions, we are looking to match their characteristics or moments. 

Moment Matching Networks: Maximum Mean Discrepancy Loss.








Minimizing MMD under a universal kernel feature expansion is equivalent to minimizing a distance between all moments of the two distributions!


Issues


Deepmind - Safety demo.

Deepmind - Distributional Demo



Test Environments
Reinforcement learning is the study of teaching machines to take actions on their own. But we can’t explicitly control them.

Results

Learning Curves for Cartpole (ε-greedy vs. uncertainty exploration)
We evaluate Risk-Adaptive IQN models to three environments - Cartpole, Windy Gridworld, and Sonic the Hedgehog.

Results

Learning Curves for Maze (Zero-shot Learning vs Few-shot Learning)
We evaluate Risk-Adaptive IQN models to three environments - Cartpole, Windy Gridworld, and Sonic the Hedgehog.

Results

Learning Curves for Maze (ε-greedy vs. uncertainty exploration)
We evaluate Risk-Adaptive IQN models to three environments - Cartpole, Windy Gridworld, and Sonic the Hedgehog.

Results

Transfer
DQN
Categorical
Risk-Adaptive IQN
Cartpole
Maze Gridworld Ex.1
---
---
---
Maze Gridworld Ex.2
---
---
---
Maze Gridworld Ex.3
---
---
---
Sonic Ex. 1
Metric: Mean Reward over Episodes.
Benchmark: Vanilla DQN
We evaluate Risk-Adaptive IQN models to three environments - Cartpole, Windy Gridworld, and Sonic the Hedgehog.

Future Work and Acknowledgements

This was supported by Deep Learning Camp Jeju 2018 which was organized by TensorFlow Korea User Group. We also thank our mentors Yu-Han Liu and Taehoon Kim!


1) Compute Phase II transfer results:
Metric: Mean reward over hold-out test levels with few-shot learning. 
Environments: Maze gridworld. Sonic OpenAI Benchmark.

Contribution: Demonstrate better performance with Risk-Adaptive IQN on common transfer benchmarks.

2) Compute full safety visualizations over Phase II:
Visualize: Uncertainty distributions given challenging obstacles.
Environments: Cartpole. Maze gridworld. Sonic OpenAI Benchmark.

Contribution: Demonstrate flexible interpretability with Distributional RL Agents.





Thank You!

