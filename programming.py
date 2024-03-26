"""
Assignment 2 programming script.

* Group Member 1:
    - Name:
    - Student ID:

* Group Member 2:
    - Name:
    - Student ID:
"""


import numpy as np

from typing import Callable



# Assignmen 2: Policy Iteration & Value Iteration

def get_action_value(
    self, s:int, a:int, V:np.ndarray, gamma:float, env_transition:Callable):
    """
    Code for getting action value. Compute the value of taking action a in state s
    I.e., compute Q(s, a) = \sum_{s'} p(s'| s, a) * [r + gamma * V(s')]
    args:
        s: state
        a: action
        V: value function
        gamma: discount factor
        env_transition: transition function
    returns:
        value: action value
    """
    value = 0

    for prob, next_state, reward, done in env_transition(s, a):

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    return value



def get_max_action_value(
    self, s:int, env_nA:int, env_transition:Callable, V:np.ndarray, gamma:float):
    """
    Code for getting max action value. Takes in the current state and returns 
    the max action value and action that leads to it. I.e., compute
    a* = argmax_a \sum_{s'} p(s'| s, a) * [r + gamma * V(s')]
    args:
        s: state
        env_nA: number of actions
        env_transition: transition function
        V: value function
        gamma: discount factor
    returns:
        max_value: max action value
        max_action: action that leads to max action value
    """
    max_value = -np.inf
    max_action = -1

    for a in range(env_nA):

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    return max_value, max_action



def get_policy(
    self, env_nS:int, env_nA:int, env_transition:Callable, gamma:float, V:np.ndarray):
    """
    Code for getting policy. Takes in an Value function and returns the optimal policy
    args:
        env_nS: number of states
        env_nA: number of actions
        env_transition: transition function
        gamma: discount factor
        V: value function
    returns:
        policy: policy
    """
    policy = np.zeros(env_nS)

    for s in range(env_nS):
        max_value = -np.inf
        max_action = -1
        for a in range(env_nA):

            # ------- your code starts here ----- #

            

            # ------- your code ends here ------- #

        policy[s] = max_action

    return policy
    


def policy_evaluation(
    self, env_nS:int, env_transition:Callable, V:np.ndarray, gamma:float, theta:float, policy:np.ndarray):
    """
    Code for policy evaluation. Takes in an MDP and returns the converged value function
    args:
        env_nS: number of states
        env_transition: transition function
        V: value function
        gamma: discount factor
        theta: convergence threshold
        policy: policy
    returns:
        V: value function
    """ 

    while True:
        delta = 0
        for s in range(env_nS):

            # ------- your code starts here ----- #

            

            # ------- your code ends here ------- #

        if delta < theta:
            break

    return V
    


def policy_improvement(
    self, env_nS:int, env_nA:int, env_transition:Callable, policy:np.ndarray, V:np.ndarray, gamma:float):
    """
    Code for policy improvement. Takes in an MDP and returns the converged policy
    args:
        env_nS: number of states
        env_nA: number of actions
        env_transition: transition function
        policy: policy
        V: value function
        gamma: discount factor
    returns:
        policy_stable: whether policy is stable
        policy: policy
    """
    policy_stable = True

    for s in range(env_nS):

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    return policy_stable, policy



def value_iteration(
    self, gamma:float, theta:float, env_nS:int, env_nA:int, env_transition:Callable):
    """
    The code for value iteration. Takes in an MDP and returns the optimal policy
    and value function.
    args:
        gamma: discount factor
        theta: convergence threshold
        env_nS: number of states
        env_nA: number of actions
        env_transition: transition function
    returns:
        policy: optimal policy
        V: optimal value function 
    """
    V = np.zeros(env_nS)
    converged = False

    while not converged:
        delta = 0

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    policy = self.get_policy(env_nS, env_nA, env_transition, gamma, V)
    
    return policy, V
    


def policy_iteration(
    self, gamma:float, theta:float, env_nS:int, env_nA:int, env_transition:Callable):
    """
    Code for policy iteration. Takes in an MDP and returns the optimal policy
    args:
        gamma: discount factor
        theta: convergence threshold
        env_nS: number of states
        env_nA: number of actions
        env_transition: transition function
    returns:
        policy: optimal policy
        V: optimal value function
    """
    V = np.zeros(env_nS)
    policy = np.zeros(env_nS)
    converged = False

    while not converged:

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    return policy, V



# Assignmen 2: Q-Learning

def epsilon_greedy(self, Q, state, epsilon):
    if np.random.rand() < epsilon:

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    else:

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #



def Q_learning(
    self, alpha:float, gamma:float, theta:float, epsilon:float, env_nS:int, env_nA:int, env_transition, env, num_episodes=1000):
    """
    Q-learning algorithm.
    Args:
        gamma: discount factor
        theta: convergence threshold
        env_nS: number of states
        env_nA: number of actions
        env_transition: transition function
        num_episodes: number of episodes
    Returns:
        Q: learned Q-value function
        rewards: rewards obtained in each episode
    """
    Q = np.zeros((env_nS, env_nA))
    rewards = []

    for episode in range(num_episodes):
        env.reset()
        state = env.state_to_index(env.state)
        done = False

        while not done:

            # ------- your code starts here ----- #

            

            # ------- your code ends here ------- #

    return np.argmax(Q, axis=1), rewards


