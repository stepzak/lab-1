import decimal
import math
from typing import Callable, Dict, Union

from extra.utils import check_is_integer

def pow_validator(*args, **kwargs):
    if len(args) == 3:
        if not all([check_is_integer(x) for x in args]):
            try:
                integer_validator(args, op = "pow")
            except TypeError:
                raise TypeError("pow() 3rd argument not allowed unless all arguments are integers")

def integer_validator(*args, **kwargs):
    if isinstance(args[0], tuple):
        args = args[0]
    if not all([check_is_integer(x) for x in args]):
        args = [str(arg) for arg in args]
        raise TypeError(f"Cannot apply '{kwargs['op']}' to {', '.join(args)}: only integers are allowed")

def non_negative_validator(*args, **kwargs):
    if not all(x>=0 for x in args):
        args = [str(arg) for arg in args]
        raise TypeError(f"Cannot apply '{kwargs['op']}' to {','.join(args)}: only non-negative values are allowed")

def custom_sqrt(a: decimal.Decimal | int) -> decimal.Decimal | int:
    ret = math.sqrt(a)
    if ret.as_integer_ratio()[1]==1.0:
        return int(ret)
    return decimal.Decimal(ret)

OPERATORS: Dict[str, tuple[float, Callable[[decimal.Decimal, decimal.Decimal], Union[decimal.Decimal, int]], bool, list[Callable] | None]] = {
        "+": (0, lambda x, y: x + y,  False, None),
        "-": (0, lambda x, y: x - y, False, None),
        "*": (1, lambda x, y: x * y, False, None),
        "/": (1, lambda x, y: x / y, False, None),
        "//": (1, lambda x, y: x // y, False, [integer_validator]),
        "%": (1, lambda x, y: x % y, False, [integer_validator]),
        "**": (2, lambda x, y: x ** y, True, None),
        "==": (-3, lambda x, y: int(x == y), False, None),
        "!=": (-3, lambda x, y: int(x != y), False, None),
        "<": (-3, lambda x, y: int(x < y), False, None),
        "<=": (-3, lambda x, y: int(x <= y), False, None),
        ">": (-3, lambda x, y: int(x > y), False, None),
        ">=": (-3, lambda x, y: int(x >= y), False, None),
        "&": (-1, lambda x, y: int(x) & int(y), False, [integer_validator]),
        "^": (-2, lambda x, y: int(x) ^ int(y), False, [integer_validator]),
        "|": (-2, lambda x, y: int(x) | int(y), False, [integer_validator]),
    } #operator: (priority, function, is_right_assoc, validator


FUNCTIONS_CALLABLE_ENUM: dict[str, tuple[Callable, list[Callable] | None, tuple[int, int]]] = {
        "max": (max, None, (1, -1)),
        "min": (min, None, (1, -1)),
        "abs": (abs, None, (1, 1)),
        "pow": (pow, [pow_validator], (2, 3)),
        "sqrt": (custom_sqrt, [non_negative_validator], (1, 1)) #function_name: (function, validators)
    }
