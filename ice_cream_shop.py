###########################################################
#                                                         #
#  Implements the environment (theoretically...)          #
#                                                         #
###########################################################

import sys
import numpy as np


class SHOP:

    def __init__(self, queue_capacity=8, num_queues=2, scoop_cost=1, leave_loss=5, queue_probs=(0.3, 0.6)):
        self.queue_capacity = queue_capacity
        self.num_queues = num_queues
        self.scoop_cost = scoop_cost
        self.leave_loss = leave_loss
        self.queue_probs = queue_probs
        self.discount_factor = 0.9
        
        # initialize state value function
        self.state_values = dict()

        # States are stored as tuples so they can be used as keys to a dictionary.
        # Because they are tuples, they can't be resized or modified after creation.
        # To support arbitrary number of queues and queue sizes, create states as array first,
        # and then convert to a tuple. 
        # We enumerate to create every possible state by finding every combination of how many
        # people are waiting in each queue 
        for i in range(0, (queue_capacity+1) ** num_queues):
            queue_lengths = [0 for x in range(0, num_queues)]
            for q in range(0, num_queues):
                queue_lengths[q] = i % (queue_capacity+1)
                i = int(i / (queue_capacity+1))
            print(queue_lengths)
            self.state_values[tuple(queue_lengths)] = 0        # state value initialized to 0

        # the actions that can be taken are serving each queue
        self.actions = [i for i in range(0, num_queues)]

        # initialize the policy (probability of taking an action in a state)
        self.policy = {}
        for s in self.state_values:
            # p is array of probabilities of taking each action in current state
            # initialize each probability to 0
            p = [0 for i in range(0, len(self.actions))]
            
            # default policy is initialized to serve the longest queue (makes more intuitive sense than random)
            longest_queue = s.index(max(s))
            p[longest_queue] = 1

        # Need an initialized policy for comparison in policy iteration
        self.dummy_policy = self.policy

        return

    def print_shop(self):
        print(self.queue_capacity)
        print(self.num_queues)
        print(self.scoop_cost)
        print(self.leave_loss)
        print(self.queue_probs)
        print(self.state_values)
        return

    # get the average immediate reward for taking an action in a state (accounting for probabilities of each queue overflowing)
    # state (tuple) is a tuple containing the number of people in each queue
    # action (int) is the queue served
    def reward(self, state, action):
        reward = 0
        if (state[action] > 0):
            reward += 1
        
        # for each queue in the state
        for q in range(0, len(state)):
            if q != action and state[q] == self.queue_capacity:
                reward -= self.leave_loss * self.queue_probs[q]
        
        return reward

    # get the probability of transitioning from a start state to the end state given an action.
    # basically mulitply all the probabilities of someone joining or not joining each queue after a step
    def transition_probability(self, start_state, end_state, action):
        probability = 1.0

        # for each queue, if someone joins that queue, multiply probability by chance of someone joining queue,
        # else if no one joines that queue, mulitply by (1 - chance of person joining that queue)
        for q in range(0, self.num_queues):

            # if queue is full and not removing a person from it, then no matter what, the queue will remain the same 
            if (start_state[q] == self.queue_capacity and q != action):
                probability *= 1.0
            # if a person joined the queue (queue length increases when queue not served, 
            # or queue length stays same when it was served, excluding when the queue started as empty)
            elif (end_state[q] > start_state[q]) or ((start_state[q] != 0) and (action == q) and (end_state[q] == start_state[q])):
                probability *= self.queue_probs[q]
            # no one joined the queue
            else:
                probability *= (1.0 - self.queue_probs[q])

            # account for if transition is not possible (probability = 0)
            # i.e. ending qeuue has fewer people than before (and it was not the one served) or was served and drops by more than 1 person
            # or a random queue adds more than one person to it
            if (end_state[q] < start_state[q] and q != action) or (end_state[q] < start_state[q] - 1) or (end_state[q] > start_state[q] + 1) or start_state[q] > self.queue_capacity or end_state[q] > self.queue_capacity:

                # print a debug message because this would probably indicate a bug in the code
                print("transition from state", start_state, "to state", end_state, "is not possible")
                return 0

        return probability
    
    # get list of all states that you could transition to given a starting state and an optional action
    # state (tuple) is the starting state
    # action (int) is the action taken in that state (if none, iterate over transitions for all posible actions)
    # returns a list of all states that can be reached from the the given start state and action(s)
    # note: does not currently account for boundary cases if probability of someone joining a queue is 0 or 1 (so it includes some transitions that are impossible),
    # but if used in conjuction with transition_probability(), it will find that probability of transitioning to those states is 0%
    def state_transitions(self, state, action=None):
        possible_states = list()
        
        actions = []
        if action is None:
            actions = self.actions
        else:
            actions = [action]

        for a in actions:
            for i in range(0, int(2 ** self.num_queues)):
                next_state = list(state)
                if next_state[a] > 0:
                    next_state[a] -= 1
                
                for q in range(0, self.num_queues):
                    person_arrives = i % 2
                    if person_arrives and next_state[q] < self.queue_capacity:
                        next_state[q] += 1
                    i = int(i / 2)

                state_tuple = tuple(next_state)
                if state_tuple not in possible_states:
                    possible_states.append(state_tuple)
        
        return possible_states

    # print all the transitions that can occur from each state and their probabilities
    # for debugging purposes
    def print_transitions(self):
        for s in self.state_values.keys():
            for a in self.actions:
                transitions = self.state_transitions(s, a)
                
                print("state", s, "action", a)

                for s2 in transitions:
                    p = self.transition_probability(s, s2, a)
                    print("\t", s, "->", s2, "= ", p)
    
    
    # Alex
    def value_iteration(self):
        # iterate some arbitrarily large number of times (for now)
        for i in range(0, 10000):
            # for each state
            for s in self.state_values:
                return_per_action = []
                for a in self.actions:
                    transitions = self.state_transitions(s, a)
                    expected_reward = 0
                    for s2 in transitions:
                        p = self.transition_probability(s, s2, a)
                        r = self.reward(s, a)
                        v2 = self.state_values[s2]
                        expected_reward += p * (r + v2 * self.discount_factor)
                    return_per_action.append(expected_reward)
                self.state_values[s] = max(return_per_action)
                
                new_policy = [0 for i in range(0, len(self.actions))]
                best_action = return_per_action.index(max(return_per_action))
                new_policy[best_action] = 1
                self.policy[s] = new_policy 
        
        for s, v in self.state_values.items():
            p = self.policy[s]
            print(s, ": ", v, "action = ", p.index(max(p)))

        return


    ############################# Sam ###############################

    #######################################################
    #                                                     #
    #  Function to evaluate the current policy            #
    #                                                     #
    #######################################################
    def evaluate_policy(self, error=1e-4):

        # Loop until the difference between the current state value and the
        # next update is less than the given error value

        delta = error + 1
        while(delta > error):
            delta = 0

            # Enumerate all of the states
            for s in self.state_values:
                return_per_action = []
                for a in self.actions:
                    transitions = self.state_transitions(s, a)
                    expected_reward = 0
                    for s2 in transitions:
                        p = self.transition_probability(s, s2, a)
                        r = self.reward(s, a)
                        v = self.state_values[s2]
                        expected_reward += p*(r + v*(self.discount_factor))
                    return_per_action.append(expected_reward)
                
                state_value_update = max(return_per_action)

                # Update the error
                if (delta < abs(state_value_update - self.state_values[s])):
                    delta = abs(state_value_update - self.state_values[s])
                self.state_values[s] = max(return_per_action)
        return
        

    #######################################################
    #                                                     #
    #  Function to update the policy based on evaluation  #
    #                                                     #
    #######################################################
    def update_policy(self):
        
        # Enumerate all of the states
        for s in self.state_values:
            return_per_action = []
            for a in self.actions:
                transitions = self.state_transitions(s, a)
                expected_reward = 0
                for s2 in transitions:
                    p = self.transition_probability(s, s2, a)
                    r = self.reward(s, a)
                    v = self.state_values[s2]
                    expected_reward += p*(r + v*(self.discount_factor))
                return_per_action.append(expected_reward)

            # Determine all of the best actions to take (If we want to consider more than one)
           # best_moves = []*len(self.actions)
           # for a in range(0, len(self.actions)):
           #     if return_per_action[a] == max(return_per_action):
           #         best_moves[a] = 1
           #     else:
           #         best_moves[a] = 0
           # num_moves = sum(best_moves)

            # Update the policy for state 's' based on the estimated reward values
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
                

if __name__ == '__main__':

    # no arguments given, use default settings for project
    if (len(sys.argv) == 1):
        shop = SHOP()

    # If one argument is given, assume it specifies value/policy iteration
    elif (len(sys.argv) == 2):
        shop = SHOP()
        iter_type = int(sys.argv[1])

    # else use custom settings from command line
    else:
        # TODO -- Make the command line arguments more modular...
        queue_capacity = int(sys.argv[1])
        num_queues = int(sys.argv[2])
        scoop_cost = int(sys.argv[3])
        leave_loss = abs(int(sys.argv[4]))
        queue_probs = np.array([float(sys.argv[x]) for x in range(5, len(sys.argv)-1)])
        iter_type = int(sys.argv[len(sys.argv)-1]) # policy or value iteration

        # Do error checking here...

        shop = SHOP(queue_capacity, num_queues, scoop_cost, leave_loss, queue_probs)
    
    #shop.print_shop()
    #shop.print_transitions()

    if (iter_type == 1):
        print('Value Iteration:')
        shop.value_iteration()
    else:
        print('Policy Iteration:')
        shop.policy_iteration()

    print('FINISHED')
    exit()

