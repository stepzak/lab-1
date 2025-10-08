from dataclasses import dataclass
from typing import Callable


@dataclass
class Function:
    """
    Class representing a function
    :param callable_function: function that will be called when met in expression
    :param validators: list of callable validators will be called before calling 'callable_function'
    :param min_args: min amount of args
    :param max_args: max amount of args
    """

    callable_function: Callable
    validators: list[Callable]
    min_args: int
    max_args: int

@dataclass
class FunctionPlaceholder:
    """
    placeholder for function
    """
    expression: str
    indexed_args: list[tuple[str, str | None]]
    min_args: int
    max_args: int

@dataclass
class Operator:
    """
    Class representing an operator
    :param priority: priority of operator
    :param callable_function: function that will be called when met in expression(left and right operands passed as arguments)
    :param is_right: is operator right associative
    :param validators: list of callable validators will be called before calling 'callable_function'
    """
    priority: float
    callable_function: Callable
    is_right: bool
    validators: list[Callable]

@dataclass
class OperatorPlaceholder:
    expression: str
    priority: float
    is_right: bool

@dataclass
class Variable:
    """
    Class representing a variable
    :param value: expression set after '='
    :local: is variable local(e.g. defined as argument of function)
    """
    value: str
    local: bool = False
