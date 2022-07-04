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

Input: 
 * Environment $E$
 * Policy model $M$ (e.g. DQN)
 * Reward shaping model $R_{s}$

Output:
 * Policy $\pi_s$ consistent with user feedback

WHILE True:
        $B$ =initialize_buffer$(E)$                                     \\\\ initialize buffer with environment trajectories    
        $\pi$ = train$(M, E, R_s)$                                      \\\\ learn partial policy
        summary  = summarize($\pi, E$)                                  \\\\ summarize policy
        feedback = get_feedback(summary, user)                          \\\\ gather feedback trajectories and their explanations from the user                
        for each feedback_trajectory, explanation:      
            $D_A$ = augment_feedback(feedback_trajectory, explanation)  \\\\ augment feedback            
            $B$ =  update_buffer($B, D_A$)                              \\\\ update buffer with augmented feedback     

        update_reward_model$(R_s, B)$                                   \\\\ update reward model with feedback data    
  

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

Environment reward:
* Cost for buying items
* Profit for selling
* Cost for not satisfying demand

### Limitations

## Open Questions

* How to choose summary of policy to present to the user? How to make sure presented trajectories summaries agent's learned knowledge and not random behavior?
* How to augment user's feedback using the provided explanation?
* How to learn reward shaping from augmented feedback?


