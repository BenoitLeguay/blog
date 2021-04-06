---
title: Deep Deterministic Policy Gradient
layout: post
permalink: "/rl/ddpg"
---

[Here is the repo](https://github.com/BenoitLeguay/DDPG)

Deep Deterministic Policy Gradient (DDPG) is an algorithm which  concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. 

**Results**

- Pendulum-v0

![pendulum ddpg learning]({{ site.baseurl }}/images/pendulum_ddpg_learning.png)

*Cumulative Reward per episode (rolling average window: 10)*

![pendulum ddpg actions]({{ site.baseurl }}/images/pendulum_ddpg_actions.png)

*Actions distributions*



*source: https://spinningup.openai.com/en/latest/algorithms/ddpg.html*