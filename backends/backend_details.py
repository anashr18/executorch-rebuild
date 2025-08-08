class BackendDetails:
    def __init__(self, name, version):
        self.name = name
        self.version = version

    def info(self):
        return f"Backend: {self.name}, Version: {self.version}"
