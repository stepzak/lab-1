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

SCIENTIFIC_FORM: int | None = None

MAXIMUM_DIGITS = 500_000
MAXIMUM_DIGITS_WARNING = 700_000
