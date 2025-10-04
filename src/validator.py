import string
import constants as cst
import vars
from compiler import CompiledExpression
from extra.exceptions import InvalidParenthesisError, VariableOvershadowError, InvalidTokenError

from extra.utils import CallAllMethods


class PreCompiledValidExpression(CallAllMethods):
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
        :raises InvalidVariableNameError: if vars do overshadow default function names(DFN)
        """
        expr_to_check = self.expression.replace(" =", "=")
        checks = ["let let=", "def let("]
        checks += list([f"let {dfn}=" for dfn in vars.FUNCTIONS_CALLABLE_ENUM.keys()])
        checks += list([f"def {dfn}(" for dfn in vars.FUNCTIONS_CALLABLE_ENUM.keys()])
        for check in checks:

            if check in expr_to_check:
                raise VariableOvershadowError(f"variable '{check[4:-1]}' overshadows default function name")
        return ''


class CompiledValidExpression(CallAllMethods):
    """
    Compiled expression validator.
    :param expression: expression to validate
    """
    def __init__(self, expression: CompiledExpression):
        self.expression = expression.expression
        self.AVAILABLE_SYMBOLS = ''.join(expression.var_map.keys())+''.join(vars.FUNCTIONS_CALLABLE_ENUM.keys())+''.join(vars.OPERATORS.keys())+"(),[]"+string.digits+"let;defreturn"
        self.var_map = expression.var_map
        self.func_map = expression.func_map
        self.op_map = expression.op_map
        self.call_all_methods()

    def check_forbidden(self):
        for s in self.expression:
            if s in cst.FORBIDDEN_SYMBOLS:
                raise InvalidTokenError(f"Symbol '{s}' is forbidden", exc_type="forbidden_symbol")

if __name__ == "__main__":
    print(PreCompiledValidExpression("5+(1)"))
