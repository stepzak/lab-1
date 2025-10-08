import pytest
from decimal import Decimal
from vars import OPERATORS
from src.calculator import Calculator
import extra.exceptions as exc

@pytest.mark.parametrize("expression",
                         [
                            "max(1, 4)",
                             "min(2, 3)",
                             "abs(-401+49)",
                             "max(abs(345-493*39), 98**3**2)",
                             "45*max(min(abs(-493*9-10000), 4940, 454), 3894, 100*48)"
                         ]
)
def test_basic_functions(expression):
    assert Calculator().calc(expression)==eval(expression)


@pytest.mark.parametrize("expression, expected",
                         [
                             ("let x = 5; let y = x+5; sqrt(y+ x)**2", Decimal(15).sqrt()**2)
                         ])
def test_vars_sqrt_bf(expression, expected):
    assert Calculator().calc(expression)==expected


@pytest.mark.parametrize("expression, exception",
                         [
                             ("let x =; x", SyntaxError),
                             ("let = 3; ", SyntaxError),
                             ("let x 5; ", SyntaxError),
                             ("let x= 5 let y = 6; x+y", SyntaxError),
                         ])
def test_invalid_lets(expression, exception):
    with pytest.raises(exception):
        Calculator().calc(expression)


@pytest.mark.parametrize("expression, exception",
                         [
                             ("sqrt(-1)", TypeError),
                             ("sqrt()", TypeError),
                             ("sqrt(1, 2)", TypeError),
                             ("max()", TypeError),
                             ("min()", TypeError),
                             ("abs(1, -3)", TypeError),
                             ("pow(1, -3.1, 3)", TypeError),
                         ])
def test_invalid_args(expression, exception):
    with pytest.raises(exception):
        Calculator().calc(expression)

@pytest.mark.parametrize("expression, exception",
                         [
                             (f"let x{op} = 5;", exc.InvalidTokenError) for op in OPERATORS.keys() if op not in ["==", "!="]
                         ])
def test_invalid_operators(expression, exception):
    with pytest.raises(exception):
        Calculator().calc(expression)
