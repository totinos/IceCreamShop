###########################################################
#                                                         #
#  Implements the environment (theoretically...)          #
#                                                         #
###########################################################

import sys
import numpy as np


class SHOP:

    def __init__(self, queue_capacity, num_queues, scoop_cost, leave_loss, queue_probs):
        self.queue_capacity = queue_capacity
        self.num_queues = num_queues
        self.scoop_cost = scoop_cost
        self.leave_loss = leave_loss
        self.queue_probs = queue_probs
        
        # initialize state value function
        self.state_values = dict();

        # States are stored as tuples so they can be used as keys to a dictionary.
        # Because they are tuples, they can't be resized or modified after creation.
        # To support arbitrary number of queues and queue sizes, create states as array first,
        # and then convert to a tuple. 
        # We enumerate to create every possible state by finding every combination of how many
        # people are waiting in each queue 
        for i in range(0, queue_capacity ** num_queues):
            queue_lengths = [0 for x in range(0, queue_capacity)];
            for q in range(0, num_queues):
                queue_lengths[q] = i % queue_capacity;
                i = int(i / queue_capacity);
            print(queue_lengths);
            self.state_values[tuple(queue_lengths)] = 0;        # state value initialized to 0

        # the actions that can be taken are serving each queue
        self.actions = [i for i in range(0, num_queues)];

        return

    def print_shop(self):
        print(self.queue_capacity)
        print(self.num_queues)
        print(self.scoop_cost)
        print(self.leave_loss)
        print(self.queue_probs)
        print(self.state_values);
        return

    # get the average immediate reward for taking an action in a state (accounting for probabilities of each queue overflowing)
    # state (tuple) is a tuple containing the number of people in each queue
    # action (int) is the queue served
    def reward(self, state, action):
        reward = 0;
        if (state[action] > 0):
            reward += 1;
        
        for q in state:
            if q != a and state[q] == self.queue_capacity:
                reward -= 5 * self.queue_probs[q];
        
        return reward;

    # get the probability of transitioning from a start state to the end state given an action.
    # basically mulitply all the probabilities of someone joining or not joining each queue after a step
    def transition_probability(self, start_state, end_state, action):
        probability = 1.0;

        # for each queue, if someone joins that queue, multiply probability by chance of someone joining queue,
        # else if no one joines that queue, mulitply by (1 - chance of person joining that queue)
        for q in range(0, self.num_queues):
            if end_state[q] > start_state[q] or (action == q and end_state[q] == start_state[q]):
                probability *= self.queue_probs[q];
            else:
                probability *= (1.0 - self.queue_probs[q]);

            # account for if transition is not possible (probability = 0)
            # i.e. ending qeuue has fewer people than before (and it was not the one served) or was served and drops by more than 1 person
            # or a random queue adds more than one person to it
            if (end_state[q] < start_state[q] and q != a) or (end_state[q] < start_state[q] - 1) or (end_state[q] > start_state[q] + 1):

                # print a debug message because this would probably indicate a bug in the code
                print("transition from state", start_state, "to state", end_state, "is not possible");
                return 0;

        return probability;

if __name__ == '__main__':

    # TODO -- Make the command line arguments more modular...
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    c = int(sys.argv[3])
    d = int(sys.argv[4])
    e = np.array([float(sys.argv[5]), float(sys.argv[6])])
    
    # Do error checking here...

    shop = SHOP(a, b, c, d, e)
    shop.print_shop()


