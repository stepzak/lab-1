from typing import Literal


class InvalidParenthesisError(Exception):
    def __init__(self, message, exc_type: Literal["empty", "unbalanced"]):
        super().__init__(message)
        self.exc_type = exc_type


class VariableOvershadowError(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidTokenError(Exception):
    def __init__(self, message, exc_type: Literal["forbidden_symbol", "unknown_token", "invalid_token"]):
        super().__init__(message)
        self.exc_type = exc_type
