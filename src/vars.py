import decimal
import math

from extra.types import Operator, Function
from extra.utils import check_is_integer

def pow_validator(*args, **kwargs):
    if len(args) == 3:
        if not all([check_is_integer(x) for x in args]):
            try:
                integer_validator(args, op = "pow")
            except TypeError:
                raise TypeError("pow() 3rd argument not allowed unless all arguments are integers")
    if args[0] == args[1] == 0:
        raise TypeError("'0**0' is undefined")

def integer_validator(*args, **kwargs):
    """
    Validates that every number is integer
    :param args: numbers to validate
    :param kwargs: must contain 'op' key with operator
    :raises TypeError: if any of number is not an integer
    """
    if isinstance(args[0], tuple):
        args = args[0]
    if not all([check_is_integer(x) for x in args]):
        args = [str(arg) for arg in args]
        raise TypeError(f"Cannot apply '{kwargs['op']}' to {', '.join(args)}: only integers are allowed")

def non_negative_validator(*args, **kwargs):
    """
    Validates that the expression is non-negative.
    :param kwargs: must contain 'op' key with operator
    :param args: numbers to validate
    :raises TypeError if any of the arguments are negative
    """
    if any(x<0 for x in args):
        args = [str(arg) for arg in args]
        raise TypeError(f"Cannot apply '{kwargs['op']}' to {','.join(args)}: only non-negative values are allowed")

def custom_sqrt(x: decimal.Decimal | int) -> decimal.Decimal | int:
    """
    Precise calculation of square root of a number.
    :param x: number to calculate square root of
    :return: square root of x
    """
    ret = decimal.Decimal(x).sqrt()
    if ret.as_integer_ratio()[1]==1.0:
        return int(ret)
    return ret

def custom_log10(x):
    x = decimal.Decimal(x)
    return x.log10()

OPERATORS: dict[str, Operator] = {
        "+": Operator(0, lambda x, y: x + y,  False, []),
        "-": Operator(0, lambda x, y: x - y, False, []),
        "*": Operator(1, lambda x, y: x * y, False, []),
        "/": Operator(1, lambda x, y: x / y, False, []),
        "//": Operator(1, lambda x, y: x // y, False, [integer_validator]),
        "%": Operator(1, lambda x, y: x % y, False, [integer_validator]),
        "**": Operator(2, lambda x, y: pow(x, y), True, [pow_validator]),
        "==": Operator(-3, lambda x, y: int(x == y), False, []),
        "!=": Operator(-3, lambda x, y: int(x != y), False, []),
        "<": Operator(-3, lambda x, y: int(x < y), False, []),
        "<=": Operator(-3, lambda x, y: int(x <= y), False, []),
        ">": Operator(-3, lambda x, y: int(x > y), False, []),
        ">=": Operator(-3, lambda x, y: int(x >= y), False, []),
        "&": Operator(-1, lambda x, y: int(x) & int(y), False, [integer_validator]),
        "^": Operator(-2, lambda x, y: int(x) ^ int(y), False, [integer_validator]),
        "|": Operator(-2, lambda x, y: int(x) | int(y), False, [integer_validator]),
    }


FUNCTIONS_CALLABLE_ENUM: dict[str,Function] = {
        "max": Function(max, [], 1, -1),
        "min": Function(min, [], 1, -1),
        "abs": Function(abs, [], 1, 1),
        "pow": Function(pow, [pow_validator], 2, 3),
        "sqrt": Function(custom_sqrt, [non_negative_validator], 1, 1),
        "sin": Function(lambda x: decimal.Decimal(math.sin(x)), [], 1, 1)
    }
