---
layout: post
title: Simplified Intuitive Explanations for RL Concepts
categories: reinforcement_learning
author: "Felix Su"
---

### On Policy vs. Off Policy

| | On Policy | Off Policy |
| --- | --- | --- |
| Definition | Policy being used to explore the environment is the same as the policy being optimized | Policy being used to explore the environment is independent of the policy being optimized |
| Examples | SARSA<br>N-Step<br>Actor-Critic | Q-Learning<br>Evolution Strategies |

**Intuitive Explanation**

The key here is the difference between a "behavior policy" and a "target policy" for an RL agent. The *target policy* is the policy that we want to optimize and learn, while the *behavior policy* is the method by which we choose what the agent does during exploration. In off-policy methods, the behavior policy simply **supplies us with information** needed to update the target policy, while in on-policy methods, the **behavior and target policies are one and the same**.

Let's take a look at the difference between Q-Learning and SARSA as our resident off-policy and on-policy methods respectively. Imagine yourself as a blind person in a room full of obstacles and you want to find the door. Each timestep, you feel around your surroundings to determine your state and have a choice to take one action step in any direction.

Let's focus on the one difference between Q-Learning and SARSA: the max operator in front of the Q-value function in the update equation. For SARSA, the Q-value we use for updates comes from **sampling the next action $$a'$$ from our target policy** (remember that the behavior and target policies are the same for on-policy methods). For Q-Learning, we use the **maximum Q-value over all possible "next actions"**. Thus, our behavior policy is **greedy**, while our target policy may not be. *not* the same as using a Q-value that resulted from the action we sampled from our target policy. 

Because for on-policy methods, the behavior and target policies are the same, for SARSA, at each new state, we simply update our policy using the reward for that state and the Q-value of the state action pair we get from sampling from our policy again.

For an off-policy method, like Q-Learning, your **behavior policy** is different than your **target policy**. The max operator essentially gives the target policy the ability to update given the **best action** in the agent's action space, given its current space. Note that this policy is greedy whereas the target policy being updated is 

As you build up knowledge via your *behavior policy* about which actions will incur the highest reward knowledge about given states that you have previously observed, you start to be able to optimize your *target policy* about how to get across the room.

<div style="text-align:center">
	<img src="/assets/img/on-off-policy.png" width="75%" alt="Q Learning vs SARSA">
</div>

### Model-based vs Model-free

| Model Based | Model Free |
| --- | --- |
| Policy is learned given *knowledge of the state* and actions | Policy is optimized via rewards from trial and error given *no visibility into the environment (black box)*|