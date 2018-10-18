
###########################################################
#                                                         #
#  Implements the environment (theoretically...)          #
#                                                         #
###########################################################

import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class SHOP:
    
    ###########################################################
    #                                                         #
    #  Initializes the problem and sets up the state values   #
    #  and policy structures                                  #
    #                                                         #
    ###########################################################
    def __init__(self, queue_capacity=8, num_queues=2, scoop_cost=1, leave_loss=5, queue_probs=[0.3, 0.6], discount_rate=0.9):
        
        # Set up the given problem parameters
        self.queue_capacity = queue_capacity
        self.num_queues = num_queues
        self.scoop_cost = scoop_cost
        self.leave_loss = leave_loss
        self.queue_probs = np.array(queue_probs)
        self.discount_rate = discount_rate
        
        # Enumerate all of the states for the problem and initialize the value to 0
        self.state_values = {}
        for i in range(0, (self.queue_capacity+1) ** self.num_queues):
            queue_lengths = [0 for x in range(0, self.num_queues)]
            for q in range(0, self.num_queues):
                queue_lengths[q] = i % (self.queue_capacity+1)
                i = int(i / (self.queue_capacity+1))
            self.state_values[tuple(queue_lengths)] = 0
        
        # The total number of actions is the same as the number of queues
        self.actions = [a for a in range(self.num_queues)]

        # Initialize the policy (probability of taking an action in a given state)
        self.policy = {}
        for s in self.state_values:
            p = np.ones((len(self.actions))) * (1 / len(self.actions))
            self.policy[s] = p

        # Need an initialized policy for comparison in policy iteration
        self.dummy_policy = self.policy

        # A list to store the policies at all iterations
        self.policy_evolution = []

        return


    ###########################################################
    #                                                         #
    #  Prints information about the current problem state     #
    #                                                         #
    ###########################################################
    def print_shop(self):
        
        print('-------------------------------------------')
        print('STATE: STATE VALUE -- BEST ACTION -- POLICY')
        print('-------------------------------------------')
        for s in self.policy:
            print('{0}: {1:6f} -- {2} -- {3}'.format(s, self.state_values[s], np.argmax(self.policy[s]), self.policy[s]))
        print('-------------------------------------------')
        return


    ###########################################################
    #                                                         #
    #  Plots the value function for each state as the number  #
    #  people in one queue vs. the number of (only for 2d)    #
    #                                                         #
    ###########################################################
    def plot_value_function(self):
        
        val = np.zeros((self.queue_capacity+1, self.queue_capacity+1))
        for i in range(len(val)):
            for j in range(len(val[i])):
                state = (i, j)
                val[i][j] = self.state_values[state]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_data, y_data = np.meshgrid(np.arange(val.shape[1]), np.arange(val.shape[0]))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = val.flatten()
        ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
        ax.set_title('Optimal Value Function for 2 Queues of 8 Capacity')
        ax.set_xlabel('Size of Queue 1')
        ax.set_ylabel('Size of Queue 2')
        ax.set_zlabel('Value of State (x, y)')
        plt.savefig('ValueFunction.png')


    ###########################################################
    #                                                         #
    #  Calculates the average immediate reward for taking a   #
    #  given action from a given state                        #
    #                                                         #
    ###########################################################
    def reward(self, state, action):
        reward = 0
        
        # If someone is served, add a reward of +1
        if (state[action] > 0):
            reward += 1
        
        # If a full queue is not served, add negative reward * overflow probability
        for q in range(0, len(state)):
            if (q != action and state[q] == self.queue_capacity):
                reward -= self.leave_loss * self.queue_probs[q]
        
        return reward


    ###########################################################
    #                                                         #
    #  Determines the probability of transitioning from a     #
    #  start state to an end state given an action            #
    #                                                         #
    ###########################################################
    def get_transition_probability(self, start_state, end_state, action):
        probability = 1.0

        # For each queue, multiply by queue_probs[q] if someone joined, and
        # multiply by (1 - queue_probs[q]) if no one joined
        for q in range(0, self.num_queues):
            
            # Checks for impossible state transitions and returns 0.0 if found
            if ((q != action and end_state[q] < start_state[q]) or
                (end_state[q] < start_state[q] - 1) or
                (end_state[q] > start_state[q] + 1) or
                (start_state[q] > self.queue_capacity) or
                (end_state[q] > self.queue_capacity)):
                print('Transition from state', start_state, 'to state', end_state, 'is not possible.')
                probability *= 0.0
                return probability

            # Probability does not change if the queue stays the same (redundant)?
            if (q != action and start_state[q] == self.queue_capacity):
                probability *= 1.0

            # Someone has joined the queue
            elif ((end_state[q] > start_state[q]) or
                (q == action and start_state[q] != 0 and end_state[q] == start_state[q])):
                probability *= self.queue_probs[q]

            # No one has joined the queue
            else:
                probability *= (1.0 - self.queue_probs[q])

        return probability


    ###########################################################
    #                                                         #
    #  Returns all states that you can transition to from a   #
    #  given state (and optionally a given action as well)    #
    #                                                         #
    #  If action = None, return transitions for all actions   #
    #                                                         #
    ###########################################################
    def get_state_transitions(self, state, action=None):
        possible_states = []
        
        # Create a list of all actions
        if action is None:
            actions = self.actions
        else:
            actions = [action]
        
        # Enumerate all possible state transitions
        for a in actions:
            for i in range(0, int(2 ** self.num_queues)):
                next_state = list(state)
                if next_state[a] > 0:
                    next_state[a] -= 1

                for q in range(0, self.num_queues):
                    person_arrives = i % 2
                    if (person_arrives and next_state[q] < self.queue_capacity):
                        next_state[q] += 1
                    i = int(i / 2)

                state_tuple = tuple(next_state)
                if state_tuple not in possible_states:
                    possible_states.append(state_tuple)

        return possible_states


    ###########################################################
    #                                                         #
    #  Simulates taking an action and gives its return        #
    #                                                         #
    ###########################################################
    def take_action(self, start_state, action):
        transitions = self.get_state_transitions(start_state, action)
        expected_reward = 0
        r = self.reward(start_state, action)
        for end_state in transitions:
            p = self.get_transition_probability(start_state, end_state, action)
            v = self.state_values[end_state]
            expected_reward += p * (r + self.discount_rate*v)
        return expected_reward


    ###########################################################
    #                                                         #
    #  Implements value iteration to converge on the optimal  #
    #  policy for the learning agent (the algorithm stops     #
    #  when the current policy "close enough" to the optimal  #
    #  policy                                                 #
    #                                                         #
    ###########################################################
    def value_iteration(self, error=1e-4):
        delta = error + 1
        iter_count = 0
        while (delta > error):
            delta = 0

            self.policy_evolution.append(self.policy.copy())
            #if (iter_count%10 == 0):
            #    self.print_shop()

            for s in self.state_values:
                return_per_action = []
                for a in self.actions:
                    return_per_action.append(self.take_action(s, a))

                # Completely greedy policy update
                #new_policy = [0 for i in range(0, len(self.actions))]
                #best_action = return_per_action.index(max(return_per_action))
                #new_policy[best_action] = 1
                #self.policy[s] = new_policy

                # Policy update giving equal weight to equal actions
                new_policy = [0 for i in range(0, len(self.actions))]
                return_per_action = np.array(return_per_action)
                return_per_action = np.around(return_per_action, decimals=4)
                best_actions = (return_per_action == np.amax(return_per_action))
                num_best = np.sum(best_actions)
                for a in self.actions:
                    if (best_actions[a] == 1):
                        new_policy[a] = 1 / num_best
                self.policy[s] = new_policy

                # Update the error
                state_value_update = max(return_per_action)
                if (delta < abs(state_value_update - self.state_values[s])):
                    delta = abs(state_value_update - self.state_values[s])
               
                # Update the value function
                self.state_values[s] = state_value_update

            iter_count += 1

        print('Iterations:', iter_count)
        #print(self.policy_evolution)
        return


    ###########################################################
    #                                                         #
    #  Evaluates the current policy by iterating to converge  #
    #  to the optimal value function                          #
    #                                                         #
    ###########################################################
    def evaluate_policy(self, error=1e-4):
        delta = error + 1
        iter_count = 0
        while (delta > error):
            delta = 0
            for s in self.state_values:
                return_per_action = []
                for a in self.actions:
                    return_per_action.append(self.take_action(s, a))
                state_value_update = max(return_per_action)

                # Update the error
                if (delta < abs(state_value_update - self.state_values[s])):
                    delta = abs(state_value_update - self.state_values[s])

                # Update the value function
                self.state_values[s] = state_value_update

            iter_count += 1
        print('Iterations:', iter_count)
        return


    ###########################################################
    #                                                         #
    #  Updates the current policy based on the state values   #
    #                                                         #
    ###########################################################
    def update_policy(self):
        
        # Enumerate all states
        for s in self.state_values:
            return_per_action = []
            for a in self.actions:
                return_per_action.append(self.take_action(s, a))
            
            # Completely greedy policy update
            new_policy = [0 for i in range(0, len(self.actions))]
            best_action = return_per_action.index(max(return_per_action))
            new_policy[best_action] = 1
            self.policy[s] = new_policy

        self.policy_evolution.append(self.policy.copy())

        return


    #######################################################
    #                                                     #
    #  Function to iterate through the policy evaluation  #
    #  and policy improvement steps of policy iteration   #
    #                                                     #
    #######################################################
    def policy_iteration(self):
        
        policy_stable = False
        while (not policy_stable):
            self.evaluate_policy()
            self.update_policy()
            if (self.dummy_policy == self.policy):
                policy_stable = True
            self.dummy_policy = self.policy.copy()

        return


###########################################################
#                                                         #
#  MAIN --                                                #
#                                                         #
###########################################################
if __name__ == '__main__':

    # Doing the example setup with value iteration by default
    if (len(sys.argv) == 1):
        num_queues = 2
        queue_probs = [0.3, 0.6]
        iter_type = 0
        shop = SHOP()

    # If one arg is provided, assume it specifies value/policy iteration
    elif (len(sys.argv) == 2):
        num_queues = 2
        queue_probs = [0.3, 0.6]
        iter_type = int(sys.argv[1])
        shop = SHOP()

    # Else use command line setup Dr. Sadovnik specifies in the writeup
    else:
        queue_capacity = int(sys.argv[1])
        num_queues = int(sys.argv[2])
        scoop_cost = int(sys.argv[3])
        leave_loss = abs(int(sys.argv[4]))
        queue_probs = [float(sys.argv[x]) for x in range(5, len(sys.argv)-1)]
        iter_type = int(sys.argv[len(sys.argv)-1])
        shop = SHOP(queue_capacity, num_queues, scoop_cost, leave_loss, queue_probs)


    # TODO -- Add a policy update selector to these functions?

    # Run value iteration or policy iteration depending on value of 'iter_type'
    if (iter_type == 0):
        print('Performing Value Iteration')
        shop.value_iteration(1e-21)
    else:
        print('Performing Policy Iteration')
        shop.policy_iteration()

    # Plot the value function only if the problem is 2-dimensional
    if (num_queues == 2):
        shop.plot_value_function()

    # Print the state of the ice cream shop as output
    shop.print_shop()





    # Create a SHOP object
    #shop = SHOP(8, 3, 1, 5, [0.6, 0.3, 0.9], 0.9)
    #shop = SHOP(8, 2, 1, 5, [0.6, 0.3], 0.9)
    #shop = SHOP(8, 2, 1, 5, [1, 0.3], 0.9)
    
    
    # This part of the script shows the improvement of the policy over time.
    # It simulates the problem for 'n' time steps and plots the total reward
    # accumulated using each policy

    n = 100
    profits = np.zeros(len(shop.policy_evolution))
    policy_count = 0
    for p in shop.policy_evolution:

        if (policy_count == len(shop.policy_evolution)):
            break

        queues = np.zeros(num_queues)
        arrivals = np.zeros((num_queues, n), dtype=int)
        for q in range(num_queues):
            arrivals[q] = np.random.binomial(1, queue_probs[q], n)

        # Simulate n time steps
        for t in range(0, n):

            # Get action from policy 'p'
            state = tuple(queues)
            action = np.argmax(p[state])

            # If the queue is not empty, someone was served
            if (queues[action] != 0):
                profits[policy_count] += 1
                queues[action] -= 1

            # Add new people according to bernoulli processes
            for q in range(num_queues):
                if (queues[q] == 8):
                    profits[policy_count] -= 5
                else:
                    queues[q] += arrivals[q][t]

        policy_count += 1

    x = np.array([dx for dx in range(0, len(shop.policy_evolution))])
    fig = plt.figure()
    plt.plot(x, profits)
    plt.xlabel('Iteration')
    plt.ylabel('Reward After 100 Time Tteps')
    plt.title('Policy Improvement Over Time')
    plt.savefig('Profits.png')

    #plt.show()
