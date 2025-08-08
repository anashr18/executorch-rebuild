from backends.partitioner import Partitioner
from backends.simple_backend import SimpleBackend
from graph import ComputationGraph, Node

backend = SimpleBackend()
print(backend.details.info())

# Build a simple computation graph
nodes = [Node("add", 2, 3), Node("multiply", 4, 5), Node("add", 10, 20)]
graph = ComputationGraph(nodes)
# print(graph)
# Partition the graph nodes (for now, just one partition)
partitioner = Partitioner(graph.nodes)
partitions = partitioner.partition()
print("Partitions:", partitions)

# Execute each partition
for i, part in enumerate(partitions):
    subgraph = ComputationGraph(part)
    results = subgraph.run(backend)
    print(f"Partition {i} results:", results)


def subtract(a, b):
    return a - b


backend.register_op("subtract", subtract)
nodes = [Node("add", 2, 3), Node("multiply", 4, 5), Node("subtract", 10, 7)]
graph = ComputationGraph(nodes)
results = graph.run(backend)
print("Graph results with subtract:", results)


nodes = [
    Node("add", 2, 3),
    Node("multiply", 4, 5),
    Node("subtract", 10, 7),
    Node("divide", 8, 2),  # Not registered, should trigger error
]
graph = ComputationGraph(nodes)
results = graph.run(backend)
print("Graph results with error handling:", results)


graph.to_dot("graph.dot")


# nodes = [
#     Node("add", 2, 3),         # result[0] = 5
#     Node("multiply", 0, 4),    # result[1] = result[0] * 4 = 20
#     Node("subtract", 1, 7)     # result[2] = result[1] - 7 = 13
# ]
