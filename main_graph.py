from backends.simple_backend import SimpleBackend
from graph import ComputationGraph, Node

backend = SimpleBackend()
print(backend.details.info())

# Build a simple computation graph
nodes = [Node("add", 2, 3), Node("multiply", 4, 5), Node("add", 10, 20)]
graph = ComputationGraph(nodes)
results = graph.run(backend)
print("Graph results:", results)
