## Interactive Reward Shaping


 * Human feedback is used to augment the existing environment's reward.

 * **Main idea:** give human feedback on trajectory level, instead of for individual *<state, action>* pairs.

 * **Contributions:**
   * Trajectory-level feedback
   * Avoiding the assumption that human knows exact correct action in each state (e.g. continuous state/action spaces or complex behaviors)
   * Allow user to augment the feedback with explanations (e.g. why this behavior should be penalized)
   * Allows capturing more complex behavior (e.g. one lane change is not necessarily a bad action, but repeating it many times is)
   * **Application**: used for correcting reward misspecification, instead of for faster convergence 

### Trajectory Feedback   

* *In what form should human feedback be collected and how should it be integrated in updating the reward model?*
* If user marks a behavior <(s_1, a_1), ..., (s_k, a_k)> that should be penalized:
  * Arriving in s_k from s_1 should be penalized (e.g. arriving in another lane)
  * Performing actions a_1, ..., a_k should be penalized (e.g. changing speed every step)
  * Other options?

### Approach Overview

![](img/alg_flow.png)

### Open Questions:

* How to choose summary of policy to present to the user? How to make sure presented trajectories summaries agent's learned knowledge and not random behavior?
* How to augment user's feedback using the provided explanation?
* How to learn reward shaping from augmented feedback?

### Gridworld Environment

Initial exploration of the questions above.

Simple gridworld where agent can move and turn and should reach the goal. Moving receives -1 penalty while turning is free.

1. Train a DQN for 50 000 steps to partially learn the task
2. Provide summary to the user
   * Visualize 10 most successful trajectories in terms of the reward
3. Detecting turning behavior (consequence of not penalizing turning)
   * <(s_1, a_1), ..., (s_k, a_k)>
4. Encode the trajectory by concatenating the starting state of the turning trajectory and the difference between the starting and the ending state.
   * [s_1, (s_1 - s_k)]
   * Allow user to mark which features are important for the feedback
   * For turning trajectory, user can mark change in agent's position and orientation between the start and end state
5. Augment human feedback using explanations
   * By randomizing features that are not deemed important we can create multiple turning trajectories
6. Learn to distinguish turning trajectories from other random trajectories gathered in the environment
   * Random forest
7. Train policy further while including reward shaping when turning behavior is recognized.


### Evaluation Environments

 * Gridworld
   * Turning in place
 * Highway driving
   * Frequent lane changes
   * Frequent speed changes
 * Inventory management
   * Frequent orders
 * Pandemic simulator
   * Ignoring political cost of imposing restrictions
 * Glucose management
   * Ignoring cost of treatment
   * Ignoring termination cost
