class Node:
    def __init__(self, op, a, b):
        self.op = op
        self.a = a
        self.b = b


class ComputationGraph:
    def __init__(self, nodes):
        self.nodes = nodes

    def run(self, backend):
        results = []
        for node in self.nodes:
            if node.op == "add":
                results.append(backend.add(node.a, node.b))
            elif node.op == "multiply":
                results.append(backend.multiply(node.a, node.b))
            else:
                results.append(None)
        return results
