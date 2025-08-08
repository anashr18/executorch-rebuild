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
            op_func = backend.get_op(node.op)
            if not op_func:
                results.append(f"Error: Unknown op '{node.op}'")
                continue
            try:
                results.append(op_func(node.a, node.b))
            except Exception as e:
                results.append(f"Error: {e}")
        return results

    def to_dot(self, filename="graph.dot"):
        with open(filename, "w") as f:
            f.write("digraph ComputationGraph {\n")
            for i, node in enumerate(self.nodes):
                f.write(f'  node{i} [label="{node.op}({node.a},{node.b})"];\n')
                if i > 0:
                    f.write(f"  node{i-1} -> node{i};\n")
            f.write("}\n")
        print(f"DOT file written to {filename}")

    # def run(self, backend):
    # results = []
    # for node in self.nodes:
    #     def resolve(x):
    #         if isinstance(x, int) and x < len(results):
    #             return results[x]
    #         return x
    #     op_func = backend.get_op(node.op)
    #     if not op_func:
    #         results.append(f"Error: Unknown op '{node.op}'")
    #         continue
    #     try:
    #         a_val = resolve(node.a)
    #         b_val = resolve(node.b)
    #         results.append(op_func(a_val, b_val))
    #     except Exception as e:
    #         results.append(f"Error: {e}")
    # return results
