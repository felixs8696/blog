---
layout: post
title: Common Issues with Reinforcement Learning
categories: reinforcement_learning
author: "Felix Su"
published: false
---

## Table of Contents
{:.no_toc}
* TOC
{:toc}

## Common Issues with RL and the methods that attempt to solve them

<!-- Reinforcement learning has been a hot topic for a few years now and we have reached an inflection point. While we have seen some promising results with  -->

<!-- ## Resources -->

### Issue #1:

#### Time Correlation and Non-Stationarity


During training, RL agents traditionally take sequential actions and perform per-step updates to their policies. However, this causes each update to be strongly correlated with time with respect to the history of an agent's trajectory. This causes the sequence of observed data to be non-stationary

#### Papers attempting to solve 
- [[Peng et al.] DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf){:target="_blank"}
	- **Section 6.1**: Reference state initialization (RSI)
- Experience Replay
- Asynchronous Advatange Actor Critic (A3C)
