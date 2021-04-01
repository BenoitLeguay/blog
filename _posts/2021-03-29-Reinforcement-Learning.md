---
layout: post
title: Reinforcement Learning
---

Below is listed my recent work on RL. From basic tabular SARSA to PPO, I explored Reinforcement Learning literature and implemented most of it.  

### [1) Reinforcement Learning basics](https://github.com/BenoitLeguay/Reinforcement_Learning_Basics) 

[test]({{ site.baseurl }}/Time-Series)

As a RL enthousiast I've decided to implement many of the algorithms I found in books, courses or papers.  To me, it is the best way to truly understand them.  In this repository, you will find implementation for many of the most known RL algorithms.

You will find the algorithms lists in the sub directories. I've decided to separate them into 3 classes:	



- **Policy Based Method**: *Algorithms that directly try to find the optimal policy* $$\pi^{*}(a\mid s)$$

  - REINFORCE

- **Actor-Critic Method**: 

  *Algorithms that optimize both the **value** and the **policy** functions to find the optimal policy*

  - QAC
  - A2C

- **Value Based Method**: *Algorithms that try to find the optimal policy by estimating the associated value function* $$V^*(s)$$

  - Tabular 
    - *SARSA*
    - *Q-Learning*
    - *Expected SARSA*
    - *SARSA($$\lambda$$)*
  - Tile Coding as function approximation
    - *SARSA*
    - *Q-Learning*
    - *Expected SARSA*
    - *SARSA($$\lambda$$)*
  - Neural Network as function approximation
    - *DQN*

### [2) DDQN](https://github.com/BenoitLeguay/DDQN)

##### Results

- *Acrobot-v1*

![learning acrobot]({{ site.baseurl }}/images/acrobot_learning.png)

*Cumulative Reward per episode (rolling average window: 10)*

**Average reward last 100 episodes:** -190.9

- *Cartpole-v0*

![learning cartpole]({{ site.baseurl }}/images/cartpole_learning.png)

*Cumulative Reward per episode (rolling average window: 50)*

**Average reward last 100 episodes:** 195.6

- *LunarLander-v2*

![lunarlander learning]({{ site.baseurl }}/images/lunarlander_learning.png)

*Cumulative Reward per episode (rolling average window: 10)*

**Average reward last 100 episodes:** 88.4

### [3) PPO](https://github.com/BenoitLeguay/PPO)

**Results**

- CartPole-v0

![cartpole_ppo_learning]({{ site.baseurl }}/images/cartpole_ppo_learning.png)

*Cumulative Reward per episode (rolling average window: 10)*

![cartpole_ppo_entropy]({{ site.baseurl }}/images/cartpole_ppo_entropy.png)

*action distribution entropy per actor update (rolling average window: 500)*

### [4) DDPG](https://github.com/BenoitLeguay/DDPG)

**Results**

- Pendulum-v0

![pendulum ddpg learning]({{ site.baseurl }}/images/pendulum_ddpg_learning.png)

*Cumulative Reward per episode (rolling average window: 10)*

![pendulum ddpg actions]({{ site.baseurl }}/images/pendulum_ddpg_actions.png)

*Actions distributions*

