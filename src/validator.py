import string

import vars
from compiler import CompiledExpression
from extra.exceptions import InvalidParenthesisError, VariableOvershadowError

from extra.utils import CallAllMethods


class PreCompiledValidExpression(CallAllMethods):
    def __init__(self, expression: str):
        self.expression = expression

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
        checks += list([f"let {dfn}=" for dfn in vars.FUNCTIONS])
        checks += list([f"def {dfn}(" for dfn in vars.FUNCTIONS])
        for check in checks:

            if check in expr_to_check:
                raise VariableOvershadowError(f"variable '{check[4:-1]}' overshadows default function name")
        return ''


class CompiledValidExpression(CallAllMethods):
    def __init__(self, expression: CompiledExpression):
        self.expression = expression.expression
        self.AVAILABLE_SYMBOLS = ''.join(expression.var_map.keys())+''.join(vars.FUNCTIONS)+''.join(vars.OPERATORS.keys())+"(),[]"+string.digits+"let;defreturn"
        self.var_map = expression.var_map
        self.call_all_methods()


    def _check_beginning(self):
        oper = ""
        cur_token = ""
        for i in range(len(self.expression)):
            s = self.expression[i]


            if len(oper):
                if oper+s in vars.OPERATORS.keys():
                    oper+=s
                    continue
                elif s.isdigit() or s in self.var_map.keys():
                    return
                #raise InvalidTokenError(f"Unfinished line: operation '{oper}' has no first operand",
                 #                       exc_type="invalid_token")
            if cur_token in vars.FUNCTIONS_CALLABLE_ENUM.keys() or s.isdigit() or cur_token in self.var_map.keys():
                return

            elif s in vars.OPERATORS.keys() and s not in ("+", "-"):
                oper = s
                continue
            else:
                cur_token+=s



if __name__ == "__main__":
    print(PreCompiledValidExpression("5+(1)"))
