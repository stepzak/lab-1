from dataclasses import dataclass, field
from typing import Callable
import vars


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


@dataclass
class Context:
    """
    Class representing a context
    :param variables: map from variable name to Variable dataclass
    :param functions: map from function name to Function dataclass
    :param operators: map from operator name to Operator dataclass
    """
    variables: dict[str, Variable] = field(default_factory=dict)
    functions: dict[str, Function | FunctionPlaceholder] = field(default_factory = lambda: vars.FUNCTIONS_CALLABLE_ENUM)
    operators: dict[str, Operator | OperatorPlaceholder] = field(default_factory= lambda: vars.OPERATORS)
    outer_names_buffer: list[str] = field(default_factory=list)
