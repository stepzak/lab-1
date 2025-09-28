from constants import FUNCTIONS_SYMBOLS_ENUM
from tests.test_ok import suppress_error


def test_invalid_parenthesis():
    expression = ")4+4("
    suppress_error(expression, SyntaxError)

def test_begins_with_binary():
    expression = "*5+5"
    suppress_error(expression, SyntaxError)

def test_overshadow_max():
    expression = "let max = 5; max"
    suppress_error(expression, SyntaxError)

def test_overshadow_min():
    expression = "let min = 5; min"
    suppress_error(expression, SyntaxError)

def test_overshadow_sqrt():
    expression = "let sqrt = 5; sqrt"
    suppress_error(expression, SyntaxError)


def test_overshadow_abs():
    expression = "let abs = 5; abs"
    suppress_error(expression, SyntaxError)

def test_forbiddens():
    for k in FUNCTIONS_SYMBOLS_ENUM.keys():
        expression = f"let {k} = 5; {k}"
        suppress_error(expression, SyntaxError)

def test_unknown_symbol():
    expression = "5,4"
    suppress_error(expression, SyntaxError)
