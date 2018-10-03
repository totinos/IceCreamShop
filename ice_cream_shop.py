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
        return

    def print_shop(self):
        print(self.queue_capacity)
        print(self.num_queues)
        print(self.scoop_cost)
        print(self.leave_loss)
        print(self.queue_probs)
        return


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


