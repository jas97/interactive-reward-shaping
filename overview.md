# Interactive Reward Shaping

## Introduction

 * Human feedback is used to augment the existing environment's reward.

 * **Main idea:** give human feedback on trajectory level, instead of for individual *<state, action>* pairs.

 * **Contributions:**
   * Trajectory-level feedback
   * Avoiding the assumption that human knows exact correct action in each state (e.g. continuous state/action spaces or complex behaviors)
   * Allow user to augment the feedback with explanations (e.g. why this behavior should be penalized)
   * Allows capturing more complex behavior (e.g. one lane change is not necessarily a bad action, but repeating it many times is)
   * **Application**: used for correcting reward misspecification, instead of for faster convergence 

## Related Work

### Human-in-the-Loop RL

### Preference-based RL

### Inverse RL

### Imitation RL

## Approach

### Algorithm

![](img/alg_flow.png)

### Trajectory Feedback   

* *In what form should human feedback be collected and how should it be integrated in updating the reward model?*
* If user marks a behavior <(s_1, a_1), ..., (s_k, a_k)> that should be penalized:
  * Arriving in s_k from s_1 should be penalized (e.g. arriving in another lane)
  * Performing actions a_1, ..., a_k should be penalized (e.g. changing speed every step)
  * Other options?


## Initial Experiments

### Gridworld Environment

Environment reward:
* -1 for moving step
* 0 for turning step
* 1 for reaching goal

Under this reward, agent learns to turn in place, thus keeping cummulative reward at 0.

![](img/before_gridworld.png) ![](img/after_gridworld.png)

### Highway Environment

Environment reward:
* Lane changes not penalized

![](img/before_highway.png)  ![](img/after_highway.png)

### Inventory Environment

### Limitations

## Open Questions

* How to choose summary of policy to present to the user? How to make sure presented trajectories summaries agent's learned knowledge and not random behavior?
* How to augment user's feedback using the provided explanation?
* How to learn reward shaping from augmented feedback?


