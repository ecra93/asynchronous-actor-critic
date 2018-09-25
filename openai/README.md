# A3C TensorFlow

https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/

## Background

* Stochastic Policy Pi(s) - a probability distribution over the action
  set in state s
    * Pi_a(s) - the probability of taking action a in state s
* Value Function V(s) of a policy Pi - the expected value of a state
  s, computed as the policy-weighted (Pi) average of each possible action
  multiplied by the value V(s') of the state reached by taking that action

        V(s) = E_policy [r + GAMMA * V(s')]

* Aside: note that the definition of Q(s,a) can be related to V(s')

        Q(s,a) = r + GAMMA * V(s')
            * assumes a particular action has been chosen, unlike V(s)
            * assumes some policy thereafter

* Advantage A(s,a) = Q(s,a) - V(s), the additional value of taking a
  particular action a from state s, compared to "average" (i.e. choosing
  a random action)

* Rho
    * Rho^s0, the probability distribution over the starting states
    * Rho^Pi, the probability distribution over states, when following
      the policy Pi

## Policy Gradient

* Use deep network to approximate the policy function, Pi
    * Either: (1) sample an action from the distribuition or (2) choose the
      action with the highest probability

* Define some function J(Pi), to measure "how good" a policy is. Define as
  discounteded reward of a policy Pi, averaged over all possible starting
  states s_0
    * Hard to estimate, but doens't matter, cause we only care about
      improving it
* Compute the gradient of J(Pi) (Policy Gradient Theorem)

        

## Actor-Critic,

## Parallel Agents



