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
* User gives feedback on presented *<state, action>* pairs
* Feedback can be:
  * Binary (good/bad action in the state)
  * Explanation-augmented (good/bad action + explanation, e.g. saliency maps)
  * Demonstration (showing the best action)
* Large amount of user effort

### Preference-based RL

* User is iteratively offered a choice between two trajectories and chooses a better one. 
* Limited feedback (only binary signal is recorded)

### Inverse and Imitation RL
* User demonstrates correct behavior
* User must know how to perform the task
* Requires a large amount of user effort

## Approach

### Algorithm 

Input: 
 * Environment $E$
 * Policy model $M$ (e.g. DQN)
 * Reward shaping model $R_{s}$

Output:
 * Policy $\pi_s$ consistent with user feedback

WHILE True:\
        $B$ =initialize_buffer$(E)$                                                \\\\ initialize buffer with environment trajectories\
        $\pi$ = train$(M, E, R_s)$                                                 \\\\ learn partial policy\
        summary  = summarize($\pi, E$)                                        \\\\ summarize policy\
        feedback = get_feedback(summary, user)                         \\\\ gather feedback trajectories and their explanations from the user\
        for each (feedback_trajectory, explanation):      
                $D_A$ = augment_feedback(feedback_trajectory, explanation)       \\\\ augment feedback\
                $B$ =  update_buffer($B, D_A$)                                   \\\\ update buffer with augmented feedback

        update_reward_model$(R_s, B)$                                         \\\\ update reward model with feedback data
        
        
<p align="center"> 
  <img src="img/alg_flow.png" width="600" class="center">
<p>
  
At the beginning, only environment's reward is available and reward shaping model $R_s$ is initialized to assign 0 to each passed trajectory.

At each iteration, policy is trained for a set number of time steps using both environment's reward and reward shaping provided by $R_s$. 
  
Then agent provides a summary of its policy, and user can mark trajectories that are undesirable. User can also specify a type of feedback (e.g. outcome or action-based) and provide explanation for their decision (at the moment in the form of important features for outcome-based feedback). 
  
A dataset of trajectories similar to the one user marked are generated using the provided explanation (at the moment by randomizing unimportant features). 
  
The augmented dataset is then used to learn a supervised learning model to distinguish between undesirable and unmarked trajectories. The supervised learning model takes as an input an encoding of a trajectory and outputs a reward (at the moment 0 or -1). 
  
During training (train($M, E, R_s$)), reward shaping model is used to augment the environment's reward function in the following way:
1. At timestep $t$ agent receives environment reward $r_e(s_t, a_t)$ 
2. Agent's previous trajectory $T_p = <(s_{t-k}, a_{t-k}), ..., (s_{t-1}, a_{t-1}), (s_t, a_t)>$ is recorded. Parameter $k$ is the time horizon.
3. Reward shaping model takes as an input the trajectory $T_p$ and outputs a reward augmentation $r_s$
4. Final reward for the agent is $r = r_e + \lambda * r_s$ where $\lambda$ is a shaping parameter determining how influential the shaping is.  
  
  
### Trajectory Feedback   

* In what form should human feedback be collected and how should it be integrated in updating the reward model?
* If user marks a behavior $<(s_1, a_1), ..., (s_k, a_k)>$ that should be penalized:
  * **Outcome-based:** 
    * Arriving in s_k from s_1 should be penalized (e.g. arriving in another lane)
    * **Initial solution**: 
      * **Trajectory encoding**: $[s_1 + \Delta(s_k, s_1)]$ to capture the difference between starting and ending state. 
      * **Explanation**: offered as a set of important features
      * **Data augmentation**: augmented samples are obtained from the feedback trajectory by randomizing unimportant features.
  * **Action-based** 
    * Performing actions $a_1, ..., a_k$ should be penalized (e.g. changing speed every step)
    * **Initial solution:**
      * **Trajectory encoding:** a list of actions $[a_1, ..., a_k]$
      * **Explanation:** NOT IMPLEMENTED
      * **Data augmentation**: a neighbourhood of action sequences is generated by perturbing the feedback encoding. The most similar action sequences are chosen depending using the Dynamic Time Warping similarity to the feedback action sequence.
  * **Feature-based**
    * NOT IMPLEMENTED
  * Other options?


## Initial Experiments

### Gridworld Environment

Environment reward:
* -1 for moving step
* 0 for turning step
* 1 for reaching goal

Under this reward, agent learns to turn in place, thus keeping cummulative reward at 0. 

For initial experiment, a trajectory where agent makes 4 turns is marked as undesirable.
  
Outcome-based feedback was implemented and change in position and orientation have been marked as important features. 

**Trajectory examples:**

State = [agent's x coordinate, agent's y coordinate, goal's x coordinate, goal's y coordinate, orientation]\
Actions: 0 - move, 1 - turn

<p align="center"> 
   <img src="img/gridworld_trajectory_1.png" width="200" class="center">
   <img src="img/gridworld_trajectory_2.png" width="200" class="center">
<p>

**Results:** action distribution through successful episodes before vs after reward shaping:
  
<p align="center"> 
   <img src="img/before_gridworld.png" width="480" class="center">
   <img src="img/after_gridworld.png" width="480" class="center">
<p>

### Highway Environment

Environment reward:
* Lane changes not penalized
  
Agent trained only on environment's reward will not hesitate to change lanes.
  
For initial experiment, trajectories where agent has changed a lane have been marked as undesirable.
  
Outcome-based feedback was used and change in y position of the vehicle has been marked as an important feature. (Only lane changes which end up in different lane were marked)

**Trajectory examples**: 

State = [presence (always 1), x location, y location, x speed, y speed]\
Actions: 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER

<p align="center"> 
  <img src="img/highway_trajectory.png" width="300" class="center">
<p>

**Results** distribution of agents y-axis position before vs. after reward shaping:

 <p align="center"> 
  <img src="img/before_highway.png" width="480" class="center">
  <img src="img/after_highway.png" width="480" class="center">
<p>

### Inventory Environment

Environment reward:
* Cost for buying items
* Profit for selling
* Cost for not satisfying demand

Results in policy that orders 20 - 40 items each time step, as demand is sampled around 30:

<p align="center"> 
   <img src="img/env_reward_inventory.png" width="600" class="center">
<p>

* Assume there is a one-off fee associated with each delivery (regardless of number of items).
Optimal policy would prefer to order less often, but in bigger batches.

* For initial experiment, 5-step trajectories where agent orders 20 - 40 items are marked as undesirable.
* Action-based feedback 

**Trajectory example:**:

State = [inventory]\
Actions: order $\in [0, 100]$

<p align="center"> 
   <img src="img/inventory_trajectory.png" width="200" class="center">
<p>

* **Results**: Action distribution before reward shaping vs. after reward shaping:
  
<p align="center"> 
   <img src="img/before_inventory.png" width="480" class="center">
   <img src="img/after_inventory.png" width="480" class="center">
<p>

### Limitations

* Two-step iteration (feedback given only once, when model is half-trained)
* Strong reward shaping signal 
* Only negative reward shaping enabled
* Random forest is used as a predictor
* Only one feedback type per task
* For action feedback, fixed time window
* Naive generation of neighbourhood of a sequence of actions
  
  
## Proposed Next Steps
 * Uniting all feedback types into one reward shaping model
 * Inverse RL can use augmented trajectories to learn reward shaping signal that can be applied to each <state, action> pair in further training

<p align="center"> 
   <img src="img/IRL_approach.png" width="600" class="center">
<p>

## Open Questions

* **Summary generation:** 
  * How to choose summary of policy to present to the user?
  * How to make sure presented trajectories summaries agent's learned knowledge and not random behavior?
  * **Initial solution**: showing top 10 trajectories with respect to cumulative reward
* **Explanations**:
  * What are meaningful explanations user could give for marking a trajectory?
  * Especially in action-based feedback, what can be an explanation?
  * **Initial solution**: in outcome-based feedback, important features are provided by the user. For action-based feedback no explanations are offered so far.
* **Encoding**
  * How to encode trajectory?
  * **Initial solution:** 
    * **Outcome-based feedback**: concatenated starting state and difference between starting and ending state of the trajectory
    * **Action-based feedback**: list of actions
* **Heterodox feedback**:
  * How to integrate positive and negative feedback?
  * How to integrate different feedback types (e.g. outcome and action-based)?
  * How to integrate trajectory and state-based feedback? 
  * **Initial solution:** only negative feedback is considered and only one feedback pere task
* **Evaluation**
  * What are the baselines?
  * How many iterations should be performed? When can we stop the training? 
  * How to measure the performance of reward shaping?
