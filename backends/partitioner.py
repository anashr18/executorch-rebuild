class Partitioner:
    def __init__(self, ops):
        self.ops = ops

    def partition(self):
        # For now, just return the list of ops as a single partition
        return [self.ops]
