class SimpleBackend:
    def __init__(self):
        from backends.backend_details import BackendDetails

        self.details = BackendDetails("SimpleBackend", "0.1")

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def compile(self, ops):
        # For now, just return a string showing the ops to be compiled
        return f"Compiled ops: {ops}"
