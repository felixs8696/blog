---
layout: post
title: Stupidly Simple (yet in depth) Explanations for RL Concepts
categories: reinforcement_learning
author: "Felix Su"
---

### On Policy vs. Off Policy

| | On Policy | Off Policy |
| --- | --- | --- |
| Definition | Policy being used to explore the environment is the same as the policy being optimized | Policy being used to explore the environment is independent of the policy being optimized |
| Examples | SARSA<br>N-Step<br>Actor-Critic | Q-Learning<br>Evolution Strategies |

#### Breaking it Down

The key here is the difference between what I'll call an "action policy" and an "update policy" for an RL agent. The *action policy* is the policy that our agent samples its actions from and uses to explore. This is the policy want to optimize and eventually use for evaluation and it is described by the $$\pi(s)$$ you often see in algorithms' pseudo code. The *update policy* is the method by which we choose what state action pairs we use to update our action policy. This "update policy" (exemplified by the $$\max_{a'} Q(s',a')$$ in the Q-Learning update) is only useful in off-policy methods, which we will soon see. For on-policy methods, we just use the action policy to both do updates and choose actions.

Let's see **several examples of action policies**

*Note: These methods ensure that if enough trials are done, each action will be tried an infinite number of times, thus ensuring optimal actions are discovered.*

- $$\epsilon$$-greedy
    - With high probability $$1-\epsilon$$ the action with the highest estimated (greedy) reward is chosen. With small probability $$\epsilon$$, an action is selected uniformly at random.
- softmax
	- Sample from the action space $$\mathcal{A}$$, some $$a'$$ weighted by its action-value estimate. This is a good approach to take where the worst actions are very unfavorable.

Let's see **several examples of action policies**

*Note: These are usually algorithm-specific*

- Q-Learning update function
	- By updating Q-values using $$\max_{a'} Q(s',a')$$, the update policy learns using the best (greedy) possible state-action trajectory path even when actual exploration via the action policy is more random and not always greedy.

Let's focus on the one difference between the update functions in both Q-Learning and SARSA: the max operator in front of the Q-value that estimates our expected future reward, in other words, the "update policy" for Q-Learning.

<div style="text-align:center">
	<img src="/assets/img/q-vs-sarsa.png" width="75%" alt="Q Learning vs SARSA">
</div>
<br>

For on-policy SARSA, the Q-value we use, $$Q(a',s')$$, comes from **sampling a state action pair from our action policy** (e.g. softmax). Intuitively, by doing this, we follow state action pairs **along the known trajectory that our policy will take** and updating their Q-values, $$Q(a,s)$$, along the way. You can see that the name "on-policy" is appropriate as we are making updates by directly following our action policy.

For off-policy Q-Learning, we have an additional **update policy** (update the current Q-value using the best Q-value over all possible next actions) which is independent of our **action policy**. While we still sample an action from our action policy to physically move our agent, the state action pairs that parameterize the $$Q(s', a')$$ are chosen by a completely different **update policy**. If we maintain the same action policy for our on-policy SARSA example, the action policy is sampled, while the update policy for Q-Learning is **greedy**, making it off-policy.

Therefore, if this update policy **is different** than our action policy, then we are **not following the action policy** for our updates, giving us an "off-policy" method. If our action policy and update policies are one and the same, then we have an "on-policy method."

### Visual Example (say you don't know Q-Learning or SARSA)

Imagine you have a robot trying to find a safe path

<div style="text-align:center">
	<img src="/assets/img/on-vs-off-policy.png" width="75%" alt="On vs Off Policy Exploration">
</div>
<br>

Because off-policy algorithms such as Q-Learning have some non-deterministic exploration path (i.e. - [$$\epsilon$$-greedy, $$\epsilon$$-soft, softmax](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html#aselection){:target="_blank"})

## So which is better?

This answer, like with most RL techniques, is that it depends. The trade off can be abstracted to optimality (off-policy) vs. safety (on-policy). If taking the optimal path is extremely important and entering risky states is not too damaging (i.e. trying to get the high score in an Atari game), off-policy methods should have a higher best case performance. However, if those risky states are of high penalty (i.e. self driving vehicle crashes), on-policy methods will optimize towards the safer path.

**Relevant Topics**
- Experience Replay
- (A3C) Asynchronous Methods for Seep Reinforcement Learning

**Sources**
- Stack Overflow explanation of On vs Off Policy
	- [https://stats.stackexchange.com/a/184794](https://stats.stackexchange.com/a/184794){:target="_blank"}
- Simple Explanation of On-Policy Q-Learning and Off-Policy SARSA
	- [https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html){:target="_blank"}
- Action policies: $$\epsilon$$-greedy, $$\epsilon$$-soft, softmax
	- [https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html#aselection](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html#aselection){:target="_blank"}
- Q-Learning and SARSA Algorithm pseudo code:
	- [https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287){:target="_blank"}
- Sutton and Barto's "Reinforcement Learning: An Introduction" Book
	- [http://incompleteideas.net/book/bookdraft2017nov5.pdf](http://incompleteideas.net/book/bookdraft2017nov5.pdf){:target="_blank"}

### Model-based vs Model-free

| Model Based | Model Free |
| --- | --- |
| Policy is learned given *knowledge of the state* and actions | Policy is optimized via rewards from trial and error given *no visibility into the environment (black box)*|