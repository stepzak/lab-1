import decimal
import pytest
from src.main import calc


def suppress_error(expr: str, error):
    with pytest.raises(error):
        calc(expr)

def test_basic():
    expression = "+(1*3+2**4-4)/3%5"
    assert  calc(expression) == (1*3+2**4-4)/3%5


def test_max():
    expression = "max(1, 5)"
    assert calc(expression)==max(1, 5)

def test_min():
    expression = "min(1, 5)"
    assert calc(expression) == min(1, 5)

def test_sqrt():
    expression = "sqrt(4)"
    assert calc(expression) == decimal.Decimal(4).sqrt()

def test_pow():
    expression = "pow(4, 4, 3)"
    assert calc(expression)==pow(4, 4, 3)

def test_abs():
    expression = "abs(-4)"
    assert calc(expression) == abs(-4)

def test_vars():
    expression = "let x = 5; let xy = 5; x+xy"
    assert calc(expression) == 10

def test_vars_and_funcs():
    x = decimal.Decimal(5)
    y = decimal.Decimal(5)
    expression = f"let x = {x}; let y = {y}; abs(-max({x}+{y}, min(sqrt({x}/{y}), 3)))+pow({x}, {y}, {y})"
    assert calc(expression) == abs(-max(x+y, min((x/y)**1/2, 3)))+pow(x, y, y)
