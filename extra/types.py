from dataclasses import dataclass
from typing import Callable


@dataclass
class Function:
    callable_function: Callable
    validators: list[Callable]
    min_args: int
    max_args: int


@dataclass
class Operator:
    priority: float
    callable_function: Callable
    is_right: bool
    validators: list[Callable]


@dataclass
class Variable:
    value: str
    local: bool = False
