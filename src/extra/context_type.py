from dataclasses import dataclass, field
from extra.types import Variable, Function, Operator, FunctionPlaceholder, OperatorPlaceholder

from vars import FUNCTIONS_CALLABLE_ENUM, OPERATORS


def default_functions():
    return FUNCTIONS_CALLABLE_ENUM.copy()

def default_operators():
    return OPERATORS.copy()

@dataclass
class Context:
    """
    Class representing a context
    :param variables: map from variable name to Variable dataclass
    :param functions: map from function name to Function dataclass
    :param operators: map from operator name to Operator dataclass
    """
    variables: dict[str, Variable] = field(default_factory=dict)
    functions: dict[str, Function | FunctionPlaceholder] = field(default_factory = default_functions)
    operators: dict[str, Operator | OperatorPlaceholder] = field(default_factory= default_operators)
    outer_names_buffer: list[str] = field(default_factory=list)
