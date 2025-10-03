import pytest

from calculator import Calculator
from extra.exceptions import VariableOvershadowError, InvalidTokenError

@pytest.mark.parametrize("expression, exception",
                         [
                            ("let x = 5; def x(): return 1;", VariableOvershadowError),
                            ("operator '$': 3, ;", SyntaxError),
                            ("operator '$': 3, l+r, false; def f$(): return 1; ", InvalidTokenError),
                            ("operator '$': 3, l+r, false; let x$ = 5; ", InvalidTokenError),
                             ("operator '+':3, l-r, f;", VariableOvershadowError),
                             ("def return(): return 1;", VariableOvershadowError),
                             ("def 1(): return 1;", VariableOvershadowError),
                             ("operator 'let': 1,l+r,f; ", VariableOvershadowError),
                             ("def f(x=3, y): return 1; f(3)", ValueError),
                         ]
 )
def test_invalid_defines(expression, exception):
    with pytest.raises(exception):
        Calculator(expression).calc()

@pytest.mark.parametrize("expression, result",
                         [
                             ("def f(x, y): return x+y; f(2,5-2)", 5),
                             ("def f(x, y=3): return x -> y; operator '->': 3, l+r**2, false; f(2)", 11),
                             ("def f(x, y=3): return x -> y; operator '->': 3, l+r**2, false; f(2, 4)", 18),
                             ("let x = 5; def f(x): return x+5; f(3)+x", 13),
                             ("let x = 5; let xy = 6; operator '$': 3, l*r-16,true; operator '#': 2, (l$r)%2, false; xy#x$3 + x$xy$xy", 6)
                         ])
def test_valid_functions(expression, result):
    assert Calculator(expression).calc() == result
