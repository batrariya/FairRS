import random
import numpy as np

class BPRSampler:

    def __init__(self, interactions, num_items):
        self.interactions = interactions
        self.num_items = num_items

        self.user_pos = {}

        for u, i in interactions:
            self.user_pos.setdefault(u, set()).add(i)

        self.users = list(self.user_pos.keys())

    def sample(self, batch_size):

        users = random.sample(self.users, batch_size)

        pos_items = []
        neg_items = []

        for u in users:
            pos = random.choice(list(self.user_pos[u]))

            while True:
                neg = random.randint(0, self.num_items - 1)
                if neg not in self.user_pos[u]:
                    break

            pos_items.append(pos)
            neg_items.append(neg)

        return (
            np.array(users),
            np.array(pos_items),
            np.array(neg_items)
        )