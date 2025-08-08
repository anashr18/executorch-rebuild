class SimpleBackend:
    def __init__(self):
        from backends.backend_details import BackendDetails

        self.details = BackendDetails("SimpleBackend", "0.1")
        self.registry = {"add": self.add, "multiply": self.multiply}

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def register_op(self, name, func):
        self.registry[name] = func

    def get_op(self, name):
        return self.registry.get(name)
