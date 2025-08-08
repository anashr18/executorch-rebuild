from backends.partitioner import Partitioner
from backends.simple_backend import SimpleBackend

backend = SimpleBackend()
print(backend.details.info())
print(f"2 + 3 = {backend.add(2, 3)}")
print(f"2 * 3 = {backend.multiply(2, 3)}")

ops = ["add", "multiply", "add"]
partitioner = Partitioner(ops)
partitions = partitioner.partition()
print("Partitions:", partitions)

# Simulate compiling each partition
for i, part in enumerate(partitions):
    print(f"Partition {i}: {backend.compile(part)}")
