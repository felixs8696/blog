---
layout: post
title: "[Paper] DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"
categories: rl_papers
author: "Felix Su"
---

## Table of Contents
{:.no_toc}
* TOC
{:toc}

## Summary of the Abstract

The goal of this paper is to use well-known RL methods to train a simulated figure to copy a motion clip while also accomplishing the task shown by the motion clip (e.g. perform a back flip, throw a ball, etc.). The authors combine a motion-imitation objective, such as "copy the joint rotations of the human" with a task objective "throw the ball into a target" and train an RL agent to accomplish these objectives in simulation. They also explore methods to integrate multiple clips into the learning process and develop multi-skilled agents that can perform diverse skills in a logical sequence (e.g. roll when you need to recover from falling). They demonstrated these result with using various characters (human, Atlas robot, bipedal dinosaur, dragon) and a large variety of skills, including locomotion, acrobatics, and martial arts and achieved extremely realistic performance by the simulated RL agent.

## Two Term Reward

The combined imitation objective and task objective used for training is defined as such:

$$r_t = \omega^I r_t^I + \omega^G r_t^G$$

- Reward weights $\omega^I$ and $\omega^G$ are tunable hyperparameters
- $r_t^I$ is the **imitation reward**
- $r_t^G$ is the **task reward**

### Imitation Reward

The imitation reward is further broken down into 4 components:

$$r_t^I = =\omega^p r_t^p + \omega^v r_t^v + \omega^e r_t^e + \omega^c r_t^c$$

Where the following definitions and objectives are given:

|$r_t^p$|**Pose Reward**|Match joint orientations|
|$r_t^v$|**Velocity Reward**|Match joint velocities|
|$r_t^e$|**End-Effector Reward**|Match positions of hands and feet|
|$r_t^c$|**Center of Mass Reward**|Match center of mass positions|

#### Pose Reward

$$r_t^p =\exp\bigg[-2\bigg(\sum_j \|\hat{q}_t^j \ominus q_t^j\|^2\bigg)\bigg]$$

## Relevant Topics
- (DDPG) Deep Deterministic Policy Gradients

## Sources
- (DDPG) Continuous Control with Deep Reinforcement Learning
	- [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971){:target="_blank"}
- Daniel Takeshi: Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients
	- [https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/){:target="_blank"}
- Sutton and Barto's "Reinforcement Learning: An Introduction" Book (Chapter 13: Policy Gradient Methods)
	- [http://incompleteideas.net/book/bookdraft2017nov5.pdf](http://incompleteideas.net/book/bookdraft2017nov5.pdf){:target="_blank"}