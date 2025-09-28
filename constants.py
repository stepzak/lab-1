import decimal
import string
from typing import Callable

FORMAT = "[%(levelname)s - %(funcName)4s() ] %(message)s"
LOG_FILE = "logs/logs.log"

PRECISION = 100
ROUNDING_DIGITS = 5
ROUNDING = decimal.ROUND_HALF_UP


FUNCTIONS_SYMBOLS_ENUM: dict[str, str] = {
        "max": "@",
        "min": "#",
        "abs": "$",
        "pow": "^",
        "sqrt": "&"
    }

SYMBOLS_FUNCTIONS_ENUM: dict[str, str]  = {value: key for key, value in FUNCTIONS_SYMBOLS_ENUM.items()}

FUNCTIONS_ARGS: dict[str, tuple[int, int]] = {  #func_symbol: (min_args, max_args)
        FUNCTIONS_SYMBOLS_ENUM["max"]: (1, -1),
        FUNCTIONS_SYMBOLS_ENUM["min"]: (1, -1),
        FUNCTIONS_SYMBOLS_ENUM["abs"]: (1, 1),
        FUNCTIONS_SYMBOLS_ENUM["pow"]: (2, 3),
        FUNCTIONS_SYMBOLS_ENUM["sqrt"]: (1, 1)
    }

SYMBOLS_CALLABLE_ENUM: dict[str, Callable] = {
        FUNCTIONS_SYMBOLS_ENUM["max"]: max,
        FUNCTIONS_SYMBOLS_ENUM["min"]: min,
        FUNCTIONS_SYMBOLS_ENUM["abs"]: abs,
        FUNCTIONS_SYMBOLS_ENUM["pow"]: pow,
}

FUNCTIONS = FUNCTIONS_ARGS.keys()

AVAILABLE_SYMBOLS = string.digits+"*/+-%"+''.join(SYMBOLS_FUNCTIONS_ENUM.keys())+"().,"
