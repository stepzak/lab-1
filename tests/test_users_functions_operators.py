import pytest

from src.calculator import Calculator
import src.extra.exceptions as exc

@pytest.mark.parametrize("expression, exception",
                         [
                            ("let x = 5; def x(): return 1;", exc.VariableOvershadowError),
                            ("operator '$': 3, ;", SyntaxError),
                            ("operator '$': 3, l+r, false; def f$(): return 1; ", exc.InvalidTokenError),
                            ("operator '$': 3, l+r, false; let x$ = 5; ", exc.InvalidTokenError),
                             ("operator '+':3, l-r, f;", exc.VariableOvershadowError),
                             ("def 1(): return 1;", exc.VariableOvershadowError),
                             ("operator 'let': 1,l+r,f; ", exc.VariableOvershadowError),
                             ("def f(x=3, y): return 1; f(3)", ValueError),
                             ("operator '->': 2, l#r,false; operator '#': 2,l->r,false; 3->4", RecursionError),
                             ("def f(x): return g(x); def g(x): return f(x); f(1)", RecursionError),
                             ("let y = y+ 5; let x = y; x", RecursionError),
                         ]
 )
def test_invalid_defines(expression, exception):
    with pytest.raises(exception):
        Calculator().calc(expression)

@pytest.mark.parametrize("expression, result",
                         [
                             ("def f(x, y): return x+y; f(2,5-2)", 5),
                             ("def f(x, y=3): return x -> y; operator '->': 3, l+r**2, false; f(2)", 11),
                             ("def f(x, y=3): return x -> y; operator '->': 3, l+r**2, false; f(2, 4)", 18),
                             ("let x = 5; def f(x): return x+5; f(3)+x", 13),
                             ("let x = 5; let xy = 6; operator '$': 3, l*r-16,true; operator '#': 2, (l$r)%2, false; xy#x$3 + x$xy$xy", 84)
                         ])
def test_valid_functions(expression, result):
    assert Calculator().calc(expression) == result
