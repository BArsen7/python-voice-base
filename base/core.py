class BaseLogic:
    def __init__(self):
        self.methods = [
            name for name, value in self.__class__.__dict__.items()
            if callable(value)
        ][1:]

    def get_methods(self) -> list:
        return self.methods
