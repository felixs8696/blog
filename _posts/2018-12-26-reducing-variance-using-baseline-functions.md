---
layout: post
title: Reducing Variance Using Baseline Functions
categories: reducing_variance_in_rl_algorithms
author: "Felix Su"
comments: true
---

## Table of Contents
{:.no_toc}
* TOC
{:toc}

## Description

### Problem 1

We don't know the true expectation of our objective functions, so we sample trajectories using a Monte Carlo approach

### Problem 2

Sampling trajectories using a Monte Carlo return ($\sum_{t=0}^{T}\gamma^t r_{t+1}$) results in a high variance estimator of the expected return because the policy and environment dynamics are stochastic and chaining stochastic samples of states and actions into long trajectories can result in high variance returns.

## Using an Unbiased Baseline Function to Reduct Variance

- Assume non-discounted rewards for simplicity ($R(\tau) = \sum_{t=0}^{T-1} r_t$)

$$
\begin{align}
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
&= \mathbb{E}_{\tau \sim \pi_\theta}\bigg[\bigg(\sum_{t=0}^{T-1}r_t\bigg)\cdot \nabla_\theta\bigg(\sum_{t=0}^{T-1}\log\pi_\theta(a_t \mid s_t)\bigg)\bigg] \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\bigg[\sum_{t'=0}^{T-1}\bigg(r_{t'}\sum_{t=0}^{t'}\nabla_\theta \log\pi_\theta(a_t \mid s_t)\bigg)\bigg] \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\bigg[\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t \mid s_t)\bigg(\sum_{t'=t}^{T-1}r_{t'}\bigg)\bigg]
\end{align}
$$

1. PG = Expected return from **cumulative (full) trajectory reward** multiplied by the gradient of the log probability of taking that **entire trajectory**
2. PG = Expected return fromt he sum of the reward **at each timestep** multiplied by the sum of gradients of the log probabilities of taking the trajectory **up to said timestep**.

$$
\begin{align}
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
&= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}\bigg[\sum_{t'=0}^{T-1}r_{t'}\bigg] \\
&= \sum_{t'=0}^{T-1} \nabla_\theta \mathbb{E}_{\tau^{(t')}}\bigg[r_{t'}\bigg] \\
&= \sum_{t'=0}^{T-1} \mathbb{E}_{\tau^{(t')}}\bigg[r_{t'}\cdot \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t \mid s_t)\bigg] \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\bigg[\sum_{t'=0}^{T-1}\bigg(r_{t'}\sum_{t=0}^{t'}\nabla_\theta \log\pi_\theta(a_t \mid s_t)\bigg)\bigg]
\end{align}
$$

a. $R(\tau)=\sum_{t=0}^{T-1}r_t$
b. Linearity of Expectation
c. Same as PG equation, but trajectory is limited to timestep of current reward ($t'$) (Reminder of PG equation: $$\nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] = \mathbb{E}_{\tau\sim\pi_\theta}\bigg[R(\tau)\cdot \nabla_\theta\big(\sum_{t=0}^{T-1}\log \pi_\theta(a_t \mid s_t)\big)\bigg]$$)
d. Linearity of Expectation

Start from the log-likelihood trick

$$\nabla_\theta \mathbb{E}[f(x)] = \mathbb{E}[f(x)\nabla_\theta\log P_\theta(x)]$$

Frame in terms of some trajectory $\tau$

$$\nabla_\theta \mathbb{E}[f(\tau)] = \mathbb{E}[f(\tau)\nabla_\theta\log P_\theta(\tau)]$$

Frame function $f$ as a reward function $R$

$$\nabla_\theta \mathbb{E}[R(\tau)] = \mathbb{E}[R(\tau)\nabla_\theta\log P_\theta(\tau)]$$

Key: $R$ is simply some output given some trajectory

- Normally $R$ is the sum of rewards at each timestep over a trajectory (i.e. $\sum_{t=0}^{T-1}r_t$), which is intuitive
- Less intuitively, it can also be a **single reward** at **some** timestep $t'$ during a given trajectory where $R=(r_{t'} \mid \tau_{0:t'})$ (Still dependent on the trajectory, so it is a valid continuous function).

When $R=(r_{t'} \mid \tau_{0:t'})$:

- $R$ is only affected by the trajectory taken up to timestep $t'$

$$\therefore \nabla_\theta \mathbb{E}_{0:T}[r_{t'}] = \nabla_{0:t'}[r_{t'}]$$

$$\therefore \nabla_{\tau\sim\pi_\theta}[r_{t'}]=]\mathbb{E}_{\tau\sim\pi_\theta}\bigg[r_{t'}\cdot\nabla_\theta\bigg(\sum_{t=0}^{t'}\log\pi_\theta (a_t \mid s_t)\bigg)\bigg]$$

3. Remember the PG equation:

$$\nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] = \mathbb{E}_{\tau\sim\pi_\theta}\bigg[\underbrace{\sum_{t'=0}^{T-1}r_{t'}\cdot \nabla_\theta\big(\sum_{t=0}^{T-1}\log \pi_\theta(a_t \mid s_t)\big)}_{R(\tau)}\bigg]$$

Let $f_t := \nabla_\theta \log \pi_\theta(a_t \mid s_t)$:

$$
\left.\begin{aligned}
R(\tau)
&= r_0f_0 +\\
&= r_1f_0 + r_1f_1\\
&= \ldots \\
&= r_{T-1}f_0 + r_{T-1}f_1 + \ldots + r_{T-1}f_{T-1}
\end{aligned}\right\rbrace
\text{1st col} \\ 
\text{1st col}
$$