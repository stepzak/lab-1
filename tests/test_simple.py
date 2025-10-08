import decimal

import pytest

from src.extra.exceptions import InvalidTokenError, InvalidParenthesisError
from calculator import Calculator


@pytest.mark.parametrize(
    "expression",
    [
        "1+4",
        "10-100",
        "5*7*8",
        "5*(7-3)",
        "2**3**2",
        "2*7%4",
        "8 & 3",
        "8 ^ 3",
        "8 ^ 3 == 0",
        "9+7>18",
        "9*18<=5",
        "(2**3)**2!=512",
        "53 // 16",
        "(53+14)//16e2+5%3+(34.43+18>=60)*52.561-2**3-(21+63>4)"
    ]
)
def test_simple_ok(expression):
    assert Calculator().calc(expression) == decimal.Decimal(eval(expression))


@pytest.mark.parametrize("expression, exception",
    [
        ("5/0", ZeroDivisionError),
        ("5.3//2", TypeError),
        ("5%2.3", TypeError),
        ("2.3 & 3", TypeError),
        ("2 ^ 3.2 == 0", TypeError),
        ("2.3 | 3.2 == 0", TypeError),
    ]
)
def test_simple_invalid_operands(expression, exception):
    with pytest.raises(exception):
        Calculator().calc(expression)

@pytest.mark.parametrize("expression, exception",
    [
        ("*-5+3", InvalidTokenError),
        ("2+5*", InvalidTokenError),
        ("2 5+6", SyntaxError),
        ("x+5", InvalidTokenError),
        ("5+((3-4)", InvalidParenthesisError),
        ("(5+()6)", InvalidParenthesisError),
        (")5+5*3(", InvalidParenthesisError),
    ]
)
def test_simple_invalid_expressions(expression, exception):
    with pytest.raises(exception):
        Calculator().calc(expression)
