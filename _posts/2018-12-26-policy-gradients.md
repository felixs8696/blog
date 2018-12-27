---
layout: post
title: Policy Gradients
categories: general_rl_algorithms
author: "Felix Su"
---

## Table of Contents
{:.no_toc}
* TOC
{:toc}

### Motivation for Policy Gradients

Policy gradients perform direct gradient updates on the policy which allows for continuous or high dimensional state and action spaces. Other methods such as Q-learning methods like DQN attempt to find an approximation of the optimal policy using Q-values to represent how good each state action pair is. This means they can only handle discrete and low-dimensional action spaces. Thus policy gradient have value in end to end control tasks such as robotics.

### Outlining the Policy Gradient Exploration Environment

Since we are exploring some environment during which we take a series of steps, we define our agent's **trajectory** using state, action, reward tuples:

> $$\tau = ((s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_{T-1}, a_{T-1}, r_{T-1}))$$

For a policy gradient system, an agent executes a trajectory by choosing actions via a policy $$\pi_\theta$$ and arriving in new states by performing that action under the conditions of the environment dynamics model $$P$$. In math language this means, $$a_i \sim \pi_\theta(a_i \mid s_i)$$ and $$s_{i+1} \sim P(s_{i+1} \mid s_i, a_i)$$. *State decisions are modeled by distribution $$P$$ because taking action $$a_i$$ in the environment may not deterministically result in the expected state $$s_{i+1}$$ due to noise and other environmental factors.*

### Deriving the Policy Gradient

#### Objective Function

The whole point of this reinforcement learning method (and all RL methods for that matter) is to optimize some objective function. In our case that means simply maximizing our expected reward $R$ by optimizing our policy $\theta$.

> $$\text{maximize}_\theta \mathbb{E}_{\pi_\theta} [R]$$

There are many types of reward functions, for example a Monte Carlo reward function would look like:

> $$R = \sum_{t=0}^{T-1} \gamma^t r_t$$

*Note: The discount factor $$\gamma$$ simply uses the geometric series convergence rule to enforce the reward to be finite under as $$T \rightarrow \infty$$.*

However, we will use a generic reward function for our policy gradient derivation to keep it simple and universally applicable.

> $$R = R(\tau)$$

As with all gradient based learning methods, in order to maximize our objective function, we need to find its gradient. Once we find the gradient, we can perform gradient ascent (if you want to maximize a reward function) or gradient descent (if you want to minimize a loss function) to achieve our goal. In our case we want to perform gradient ascent on our reward function and to remind you, our objective function is: $$\text{maximize}_\theta \mathbb{E}_{\pi_\theta} [R(\tau)]$$

#### Log-Likelihood Trick

To find the gradient, we use the **log-likelihood trick**, also known as the [log-derivative trick](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/){:target="_blank"}.

$$
\begin{align*}
\nabla_\theta\mathbb{E}[R(\tau)] &= \nabla_\theta\int p_\theta(\tau) R(\tau) d\tau
&& \text{Definition of Expectation}\\
&= \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau
&& \text{Leibniz Integral Rule}^*\\
&= \int \frac{p_\theta(\tau)}{p_\theta(\tau)}\nabla_\theta p_\theta(\tau) R(\tau) d\tau
&& \text{Multiplicative Identity Property}\\
&= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau && \text{Derivative of log and Chain Rule}^{**}\\
&= \mathbb{E}[R(\tau) \nabla_\theta \log p_\theta(\tau)]
&& \text{Definition of Expectation}
\end{align*}
$$

$$^*$$ Leibniz Integral Rule [[1]](https://math.stackexchange.com/questions/2530213/when-can-we-interchange-integration-and-differentiation){:target="blank"} [[2]](https://en.wikipedia.org/wiki/Leibniz_integral_rule){:target="blank"}

$$^{**}$$ $\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta \log p_\theta(\tau)$ because $\nabla_x \log f(x) = \frac{1}{f(x)}\cdot \nabla_x f(x)$)

<mark>Up to this point, this is what we have:</mark>

**Objective Function**
> $$\text{maximize}_\theta \mathbb{E}_{\pi_\theta} [R(\tau)]$$

**Policy Gradient (unfinished)**
> $$\nabla_\theta\mathbb{E}[R(\tau)] = \mathbb{E}[R(\tau) \nabla_\theta \log p_\theta(\tau)]$$

#### Deriving Log Probability of a Trajectory

We know the reward function because it is hand-crafted, so all we now need to derive is the remaining unknown ($$\nabla_\theta \log p_\theta(\tau)$$) log probability of an entire trajectory. To do that we need the following prerequisite information.

1. Sample $$s_0$$ from initial state distribution $\mu$ where the probability of getting some state $$s_0 = \mu(s_o)$$
2. Reminder from [earlier]({{site.url}}{{page.url}}#outlining-the-policy-gradient-exploration-environment):
	- $\pi_\theta(a_t \mid s_t) =$ probability of taking action $a_t$, given state $s_t$, using policy $\pi_\theta$
	- $P(s_{t+1} \mid s_t, a_t)$ = probability of getting to state $s_{t+1}$ from state $s_t$ by taking the action $a_t$ (given by the policy), given the noisy dynamics of the environment.

$$
\begin{align*}
\nabla_\theta \log p_\theta(\tau) &= \nabla \log(\mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t)P(s_{t+1} \mid s_t, a_t)) \\
	&= \nabla_\theta \bigg[\log \mu(s_0) + \sum_{t=0}^{T-1}(\log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t))\bigg]
	&& \text{log of product = sum of logs}\\
	&= \nabla_\theta \sum_{t=1}^{T-1} \log \pi_\theta (a_t \mid s_t)
	&& \text{keep terms dependent on } \theta
\end{align*}
$$

#### Final Policy Gradient Definition
<mark>Combining all of our information together, we finally have our policy gradient defined as **the gradient of our policy over the expectation of the reward over a trajectory**</mark>

**Objective Function**
> $$\text{maximize}_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

**Final Policy Gradient**
> $$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta} \bigg[R(\tau) \cdot \nabla_\theta \bigg(\sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t)\bigg)\bigg]$$


### So which is better, Q-Learning or Policy Gradients?

Conveniently for me, this was explained very well by blogger Felix Yu in his own blog post about [DQN vs PG](https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html){:target="_blank"}. Here's his breakdown:

> Policy Gradients is generally believed to be able to apply to a wider range of problems. For instance, on occasions when the Q function (i.e. reward function) is too complex to be learned, DQN will fail miserably. On the other hand, Policy Gradients is still capable of learning a good policy since it directly operates in the policy space. Furthermore, Policy Gradients usually show faster convergence rate than DQN, but has a tendency to converge to a local optimal. Since Policy Gradients model probabilities of actions, it is capable of learning stochastic policies, while DQN can’t. Also, Policy Gradients can be easily applied to model continuous action space since the policy network is designed to model probability distribution, on the other hand, DQN has to go through an expensive action discretization process which is undesirable.

> You may wonder if there are so many benefits of using Policy Gradients, why don’t we just use Policy Gradients all the time and forget about Q Learning? It turns out that one of the biggest drawbacks of Policy Gradients is the high variance in estimating the gradient of $$E[R_t]$$. Essentially, each time we perform a gradient update, we are using an estimation of gradient generated by a series of data points $$<s,a,r,s^\prime>$$ accumulated through a single episode of game play. This is known as Monte Carlo method. Hence the estimation can be very noisy, and bad gradient estimate could adversely impact the stability of the learning algorithm. In contrast, when DQN does work, it usually shows a better sample efficiency and more stable performance.

### Reducing Variance Without Creating Bias

As noted above, a recurring issue with reinforcement learning, particularly for policy gradients, is variance in our estimations of the gradient of $$E[R(\tau)]$$. Agents have difficulty exploring and learning in a smooth, well-defined manner without inducing a biased exploration path. To prevent this post from getting too long, I have moved variance reduction methods to different blog posts. Click below to move through them.

- [Reducing Variance Using Baseline Functions]({% post_url 2018-12-26-reducing-variance-using-baseline-functions %})

### Relevant Topics
- (DDPG) Deep Deterministic Policy Gradients

### Sources
- (DDPG) Continuous Control with Deep Reinforcement Learning
	- [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971){:target="_blank"}
- Daniel Takeshi: Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients
	- [https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/){:target="_blank"}
- Sutton and Barto's "Reinforcement Learning: An Introduction" Book (Chapter 13: Policy Gradient Methods)
	- [http://incompleteideas.net/book/bookdraft2017nov5.pdf](http://incompleteideas.net/book/bookdraft2017nov5.pdf){:target="_blank"}