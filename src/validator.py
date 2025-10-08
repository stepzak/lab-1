import string
import src.constants as cst
import src.vars as vrs
from src.compiler import CompiledExpression
from src.extra.exceptions import InvalidParenthesisError, VariableOvershadowError, InvalidTokenError

from src.extra.utils import CallAllMethods


class PreCompiledValidExpression(CallAllMethods):
    """
    CLass to validate pre compiled expression
    :param expression: expression to validate
    :raises InvalidParenthesisError: invalid parenthesis
    :raises VariableOvershadowError: some of the names overshadow default ones
    """
    def __init__(self, expression: str):
        self.expression = str(expression)

        self.call_all_methods()

    def _check_parenthesis(self) -> None:
        """
        Checks if parenthesis syntax is OK
        :return: None if OK
        :raises InvalidParenthesisError otherwise
        """

        parentheses_total: int = 0

        for s in self.expression:
            if s == "(":
                parentheses_total += 1
            elif s == ")":
                parentheses_total -= 1
            if parentheses_total < 0:
                raise InvalidParenthesisError("Unbalanced parenthesis in expression", exc_type="unbalanced")
        if parentheses_total > 0:
            raise InvalidParenthesisError("Unbalanced parenthesis in expression", exc_type="unbalanced")

    def _check_vars_and_funcs(self) -> str:
        """
        Checks if vars do not overshadow default function names(DFN)
        :return: None
        :raises VariableOvershadowError: if vars do overshadow default function names(DFN)
        """
        expr_to_check = self.expression.replace(" =", "=").strip().replace(" (", "(")
        checks = ["let let=", "def let(", "def operator(", "let operator=", "def def(", "def return(", "let return=", "let def="]
        checks += list([f"let {dfn}=" for dfn in vrs.FUNCTIONS_CALLABLE_ENUM.keys()])
        checks += list([f"def {dfn}(" for dfn in vrs.FUNCTIONS_CALLABLE_ENUM.keys()])
        for check in checks:

            if check in expr_to_check:
                raise VariableOvershadowError(f"variable '{check[4:-1]}' overshadows default function name")
        return ''


class CompiledValidExpression(CallAllMethods):
    """
    Compiled expression validator.
    :param expression: expression to validate
    :raises InvalidTokenError: forbidden symbol detected
    """
    def __init__(self, expression: CompiledExpression):
        self.expression = expression.expression
        self.var_map = expression.ctx.variables
        self.func_map = expression.ctx.functions
        self.op_map = expression.ctx.operators
        self.AVAILABLE_SYMBOLS = ''.join(self.var_map.keys())+''.join(self.func_map.keys())+''.join(self.op_map.keys())+"(),[]"+string.digits+"let;defreturn"

        self.call_all_methods()

    def check_forbidden(self):
        for s in self.expression:
            if s in cst.FORBIDDEN_SYMBOLS:
                raise InvalidTokenError(f"Symbol '{s}' is forbidden", exc_type="forbidden_symbol")

if __name__ == "__main__":
    print(PreCompiledValidExpression("5+(1)"))
