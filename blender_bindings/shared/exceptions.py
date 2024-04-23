class RequiredFileNotFound(Exception):
    def __init__(self, message, *args):
        super().__init__(*args)
        self.message = message
