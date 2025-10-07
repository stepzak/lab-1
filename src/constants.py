import decimal

FORMAT = "[%(levelname)s - %(funcName)4s() ] %(message)s"
LOG_FILE = "/var/log/lab-1/logs.log"

PRECISION = 100
ROUNDING_DIGITS = 2
ROUNDING = decimal.ROUND_HALF_UP

SCIENTIFIC_FORM: int | None = 5 #5 = e5 format

MAXIMUM_DIGITS = 1_000_000
MAXIMUM_DIGITS_WARNING = 700_000

FORBIDDEN_SYMBOLS = set("[]")
NAME_FORBIDDEN_SYMBOLS = set(" (),:")
SYSTEM_NAMES = {"return", "operator", "def", "let"}
