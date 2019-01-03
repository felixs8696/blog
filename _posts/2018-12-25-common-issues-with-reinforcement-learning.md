---
layout: post
title: Common Issues with Reinforcement Learning
categories: reinforcement_learning
author: "Felix Su"
published: true
comments: true
---

## Table of Contents
{:.no_toc}
* TOC
{:toc}

## Common Issues with RL and the methods that attempt to solve them

<!-- Reinforcement learning has been a hot topic for a few years now and we have reached an inflection point. While we have seen some promising results with  -->

<!-- ## Resources -->

### Issue #1:

#### Data Correlation and Non-Stationarity

To understand **data correlation** and why it is a problem, we first have to understand that **data is generally "uniformly (equally) important"**. Even though states vary in their "reward value" (ex. a game agent like Sonic will try to avoid bad reward states like dangerous traps or beasts, and achieve good reward states like getting coins), each state holds equally important information that allows an agent to learn what to do in any situation regardless of its reward value  (i.e. If Sonic is standing in the middle of nowhere, this isn't a particularly low or high reward state, however this state should be represented equally in the data set used for updating the policy because Sonic needs to learn what to do here regardless of how often it is visited). So, ideally you want to agent to go out there, observe as many things as possible and learn from these experiences in an unbiased, "equal opportunity", kind of way. However, in reality, if an agent is being trained "online", meaning it accumulates reward along its trajectory and updates its policy using sequential data samples along the way, it will have gathered biased data samples because **states visited in the earlier parts of the trajectory affect where the agent ends up in the future**. This is what I mean by **data correlation**.

Another terrible effect of data correlation is **non-stationarity**. This means that **data correlation causes high variance in the trajectories explored and the probability distribution across state action pairs is unstable**. What this means is that **small updates to the policy function (ex. a neural network) can cause the agent to follow vastly different exploration paths**. This causes learning to have a difficult time converge. If using a neural network, you can imagine why this is the case. Gradient descent uses derivatives and the chain rule to directly update your network's weights in the direction that will definitively improve your overall reward **given the data samples used to update the network**. Read the bold part again. You see the problem here? This means that the policy is **only** updated to maximize reward given the all the state, action, rewards seen in the batch of trajectories explored by the agent before the update. However, the **resulting update will almost definitely cause the agent to explore an entirely different series of states and actions every time**. This is why so many papers reference the issues of **high variance** in reinforcement learning. Online training using consecutive data samples creates bias in exploration paths and reward collection, which in turn, causes high variance updates to the policy.

#### Papers attempting to solve 
- [[Peng et al.] DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf){:target="_blank"}
	- **Section 6.1**: Reference state initialization (RSI)
- Experience Replay
- Asynchronous Advantage Actor Critic (A3C)
