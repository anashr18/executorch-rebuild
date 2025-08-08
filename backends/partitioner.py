class Partitioner:
    def __init__(self, nodes):
        self.nodes = nodes

    def partition(self):
        partitions = {}
        for node in self.nodes:
            partitions.setdefault(node.op, []).append(node)
        # Return a list of partitions (each is a list of nodes)
        return list(partitions.values())
