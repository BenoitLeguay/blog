---
layout: post
title: Reinforcement Learning
---

**Reinforcement Learning work**

Below is listed my recent work (repositories) on RL. 

### [1) Reinforcement Learning basics](https://github.com/BenoitLeguay/Reinforcement_Learning_Basics) 

As a RL enthousiast I've decided to implement many of the algorithms I found in books, courses or papers.  To me, it is the best way to truly understand them.  

In this repository, you will find implementation for many of the most known RL algorithms.

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

**Average reward last 100 episodes:** -190.9

- *Cartpole-v0*

![learning cartpole]({{ site.baseurl }}/images/cartpole_learning.png)

**Average reward last 100 episodes:** 195.6

- *LunarLander-v2*

![lunarlander learning.png]({{ site.baseurl }}/images/lunarlander_learning.png)

**Average reward last 100 episodes:** 88.4

### [3) PPO](https://github.com/BenoitLeguay/PPO)

### [4) DDPG](https://github.com/BenoitLeguay/DDPG)

