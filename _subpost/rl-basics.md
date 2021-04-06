---
title: Reinforcement Learning basics
layout: post
permalink: "/rl/basics"
---

[Here is the repo](https://github.com/BenoitLeguay/Reinforcement_Learning_Basics)

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

