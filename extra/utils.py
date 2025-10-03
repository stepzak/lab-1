import decimal
from functools import wraps
import src.constants as cst

def check_is_integer(dec: decimal.Decimal) -> bool:
    return dec.as_integer_ratio()[1] == 1.0

def round_decimal(dec: decimal.Decimal, n_digits: int = cst.ROUNDING_DIGITS, rounding=cst.ROUNDING):
    quantizer = decimal.Decimal('1.' + '0' * n_digits)
    try:
        return dec.quantize(quantizer, rounding=rounding)
    except decimal.InvalidOperation:
        return dec

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


class CallAllMethods:
    def call_all_methods(self, instance = None):
        if not instance:
            instance = self
        for method in dir(instance):
            attr = getattr(instance, method)
            if not method.startswith("__") and callable(attr) and method!="call_all_methods":
                attr()
