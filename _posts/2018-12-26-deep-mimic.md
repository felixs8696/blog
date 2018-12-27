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

### Summary of the Abstract

The goal of this paper is to use well-known RL methods to train a simulated figure to copy a motion clip while also accomplishing the task shown by the motion clip (e.g. perform a back flip, throw a ball, etc.). The authors combine a motion-imitation objective, such as "copy the joint rotations of the human" with a task objective "throw the ball into a target" and train an RL agent to accomplish these objectives in simulation. They also explore methods to integrate multiple clips into the learning process and develop multi-skilled agents that can perform diverse skills in a logical sequence (e.g. roll when you need to recover from falling). They demonstrated these result with using various characters (human, Atlas robot, bipedal dinosaur, dragon) and a large variety of skills, including locomotion, acrobatics, and martial arts and achieved extremely realistic performance by the simulated RL agent.



#### Two Term Reward

### Relevant Topics
- (DDPG) Deep Deterministic Policy Gradients

### Sources
- (DDPG) Continuous Control with Deep Reinforcement Learning
	- [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971){:target="_blank"}
- Daniel Takeshi: Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients
	- [https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/){:target="_blank"}
- Sutton and Barto's "Reinforcement Learning: An Introduction" Book (Chapter 13: Policy Gradient Methods)
	- [http://incompleteideas.net/book/bookdraft2017nov5.pdf](http://incompleteideas.net/book/bookdraft2017nov5.pdf){:target="_blank"}