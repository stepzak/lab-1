import copy
import decimal
from functools import wraps
from types import UnionType

import src.constants as cst
import inspect
import src.vars as vrs
from src.extra.types import Context
from typing import get_args, get_origin

def check_is_integer(dec: decimal.Decimal) -> bool:
    return dec.as_integer_ratio()[1] == 1.0

def round_decimal(dec: decimal.Decimal, n_digits: int = cst.ROUNDING_DIGITS, rounding=cst.ROUNDING):
    """
    Rounds decimal to n digits after point
    :param dec: decimal to round
    :param n_digits: number of digits to round to
    :rounding: rounding method
    :return: rounded decimal
    """
    quantizer = decimal.Decimal('1.' + '0' * n_digits)
    try:
        return dec.quantize(quantizer, rounding=rounding)
    except decimal.InvalidOperation:
        return dec

def get_previous_token(tokens: list[str], index: int) -> str:
    tks = tokens[:index+1:]
    for token in tks[::-1]:
        if token.isspace():
            continue

        return token
    return tokens[0]

def get_next_token(tokens: str, index: int) -> str:
    for token in tokens[index+1:]:
        if token.isspace():
            continue
        return token
    return tokens[-1]


def log_exception(func):

    """Decorator to automatically log exceptions"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):

        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.exception(f"Exception in {func.__name__}: {e}")
            raise

    return wrapper


def init_default_ctx(cls):

    init_original = cls.__init__
    def new_init(*args, **kwargs):
        for k, v in inspect.signature(init_original).parameters.items():

            if get_origin(v.annotation) is UnionType:
                if Context in get_args(v.annotation):
                    ctx_have = kwargs.get(k, Context(functions={}, operators={}, outer_names_buffer=[], cache={}))
                    if not ctx_have.operators:
                        ctx_have.operators = vrs.OPERATORS
                    if not ctx_have.functions:
                        ctx_have.functions = vrs.FUNCTIONS_CALLABLE_ENUM
                    kwargs[k] = copy.deepcopy(ctx_have)
        return init_original(*args, **kwargs)
    cls.__init__ = new_init
    return cls


class CallAllMethods:
    """
    Calls every method of the object
    """
    def call_all_methods(self, instance = None):
        if not instance:
            instance = self
        for method in dir(instance):
            attr = getattr(instance, method)
            if not method.startswith("__") and callable(attr) and method!="call_all_methods" and method.find("__") == -1:
                attr()
