
###########################################################
#                                                         #
#  Implements the environment (theoretically...)          #
#                                                         #
###########################################################

import sys
import numpy as np


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

        return


    ###########################################################
    #                                                         #
    #  Prints information about the current problem state     #
    #                                                         #
    ###########################################################
    def print_shop(self):
        
        print()
        print('-------------------------------')
        print('POLICY:')
        print('-------------------------------')
        for p in self.policy:
            print('{0}: {1}'.format(p, self.policy[p]))

        print()
        print('-------------------------------')
        print('STATE VALUES:')
        print('-------------------------------')
        for s in self.state_values:
            print('{0}: {1}'.format(s, self.state_values[s]))
        
        print('-------------------------------')
        return


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
        # print('leave_loss:', self.leave_loss)
        # print('reward:', r)
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

        for s, v in self.state_values.items():
            p = self.policy[s]
            #print(s, ': ', v, 'action = ', p.index(max(p)))
            print(s, ': ', v, 'action = ', np.argmax(p))
        print('Iterations', iter_count)
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
        print('Iterations', iter_count)
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

        for s, v in self.state_values.items():
            p = self.policy[s]
            print(s, ": ", v, "action = ", p.index(max(p)))

        return


###########################################################
#                                                         #
#  MAIN --                                                #
#                                                         #
###########################################################
if __name__ == '__main__':

    shop = SHOP(8, 2, 1, 5, [0.3, 0.6], 0.9)
    
    #shop.print_shop()

    # Statements to help with debugging and testing
    #state = (8,7)
    #print(state)
    #print(shop.get_state_transitions(state, 1))
    #print(shop.take_action(state, 1))

    # Run value iteration
    shop.value_iteration()
    
    #shop2 = SHOP(8, 2, 1, 5, [0.3, 0.6], 0.9)
    #shop2.policy_iteration()

