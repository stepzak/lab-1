import decimal
from os import path

FORMAT = "[%(levelname)s - %(funcName)4s() ] %(message)s"
LOG_FILE =  path.join(
    path.dirname(
        path.dirname(path.abspath(__file__))
    ),
    "logs",
    "logs.log"
)

PRECISION = 125
ROUNDING_DIGITS = 2
ROUNDING = decimal.ROUND_HALF_UP
