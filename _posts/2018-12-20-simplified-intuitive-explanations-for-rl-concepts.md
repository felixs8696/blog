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

The key here is the difference between what I'll call an "action policy" and an "update policy" for an RL agent. The *action policy* is the policy that our agent samples its actions from and uses to explore. This is the policy want to optimize and eventually use for evaluation and it is described by the $$\pi(s)$$ you often see in algorithms' pseudocode. The *update policy* is the method by which we choose what state action pairs we use to update our action policy. This "update policy" (exemplified by the $$\max_{a'} Q(s',a')$$ in the Q-Learning update) is only useful in off-policy methods, which we will soon see. For on-policy methods, we just use the action policy to both do updates and choose actions.

Let's focus on the one difference between the update functions in both Q-Learning and SARSA: the max operator in front of the Q-value that estimates our expected future reward, in other words, the "update policy" for Q-Learning.

<div style="text-align:center">
	<img src="/assets/img/on-off-policy.png" width="75%" alt="Q Learning vs SARSA">
</div>
<br>

For on-policy SARSA, the Q-value we use, $$Q(a',s')$$, comes from **sampling a state action pair from our action policy** (e.g. sample from the action space $$\mathcal{A}$$, some $$a'$$ weighted by its Q-value, $$Q(a', s')$$). Intuitively, by doing this, we follow state action pairs **along the known trajectory that our policy will take** and updating their Q-values, $$Q(a,s)$$, along the way. You can see that the name "on-policy" is appropriate as we are making updates by directly following our action policy.

For off-policy Q-Learning, we have an additional **update policy** (update the current Q-value using the best Q-value over all possible next actions) which is independent of our **action policy**. While we still sample an action from our action policy to physically move our agent, the state action pairs that parameterize the $$Q(s', a')$$ are chosen by a completely different **update policy**. If we maintain the same action policy for our on-policy SARSA example, the action policy is sampled, while the update policy for Q-Learning is **greedy**, making it off-policy. 

Therefore, if this update policy **is different** than our action policy, then we are **not following the action policy** for our updates, giving us an "off-policy" method. If our action policy and update policies are one and the same, then we have an "on-policy method."

### Visual Example (say you don't know Q-Learning or SARSA)

--- TBA ---
<!-- Imagine yourself as a blind person in a room full of obstacles and you want to find the door. Each timestep, you feel around your surroundings to determine your state and have a choice to take one action step in any direction. If you take an on-policy approach, you might decide that your **action policy** is to  -->

**Relevant Topics**
- Experience Replay
- (A3C) Asynchronous Methods for Seep Reinforcement Learning


### Model-based vs Model-free

| Model Based | Model Free |
| --- | --- |
| Policy is learned given *knowledge of the state* and actions | Policy is optimized via rewards from trial and error given *no visibility into the environment (black box)*|