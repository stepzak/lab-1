import decimal
import logging
import constants as cst
from sys import stdout


logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.DEBUG,
    handlers=[
        logging.FileHandler(cst.LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(stdout),
    ],
    format = cst.FORMAT
)

def check_parenthesis(expression: str) -> bool:
    """
    Checks if parenthesis syntax is OK
    :param expression: expression to check
    :return: True if OK else False
    """
    parentheses_total: int = 0

    for s in expression:
        if s == "(":
            parentheses_total += 1
        elif s == ")":
            parentheses_total -= 1
        if parentheses_total < 0:
            return False
    res = parentheses_total == 0
    return res

def check_vars(expression: str) -> str:
    """
    Checks if vars do not overshadow default function names(DFN)
    :param expression: expression to check. Syntax: let x = ...; let y = ...; x+y
    :return: '' if vars do not overshadow DFN else name of overshadowed function
    """
    checks = ("let max=", "let min=", "let abs=", "let sqrt=", "let pow=")
    for check in checks:

        if check in expression:
            return check[4:-1]
    return ''


def check_for_forbidden_symbols(expression: str) -> str:
    """
    Checks for forbidden symbols(AKA functions replacements if src.constants)
    :param expression: expression to check
    :return: '' if no forbidden symbols detected, otherwise - the detected symbol
    """
    for k in ''.join(cst.SYMBOLS_FUNCTIONS_ENUM.keys())+'[]':
        if k in expression:
            return k
    return ''

def check_is_integer(dec: decimal.Decimal) -> bool:
    return dec.as_integer_ratio()[1]==1.0
