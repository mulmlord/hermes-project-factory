class TemplateSyntaxError(Exception):
    def __init__(self, message, line):
        super().__init__(f"line {line}: {message}")
        self.line = line


class UndefinedError(Exception):
    pass
