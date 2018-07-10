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




