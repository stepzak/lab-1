import decimal
import logging
import string
import sys
from decimal import getcontext
from sys import stdout
from typing import Any, Union

import constants as cst # type: ignore
from extra.exceptions import InvalidTokenError, InvalidParenthesisError  # type: ignore
from extra.utils import log_exceptions, round_decimal, check_is_integer #type: ignore
from validator import PreCompiledValidExpression, CompiledValidExpression  #type: ignore
from compiler import CompiledExpression #type: ignore
import vars #type: ignore
from math import log10, floor

getcontext().prec = cst.PRECISION

logger = logging.getLogger(__name__)

sys.set_int_max_str_digits(cst.MAXIMUM_DIGITS)

INITIAL_VALUES = {
            "FUNCTIONS_CALLABLE_ENUM": vars.FUNCTIONS_CALLABLE_ENUM.copy(),
            "FUNCTIONS": vars.FUNCTIONS.copy(),
            "FUNCTIONS_ARGS": vars.FUNCTIONS_ARGS.copy(),
            "OPERATORS": vars.OPERATORS.copy(),
        }

class CustomFunctionExecutor:
    def __init__(self, indexed_vars: list[tuple[str, Union[str, None]]],  expression: str):
        self.expression = expression
        self.indexed_vars = indexed_vars

    #@log_exceptions(logger = logger)
    def execute_function(self, *args, var_map: dict) -> decimal.Decimal:
        var_map = var_map.copy()
        for i in range(len(self.indexed_vars)):
            try:
                if args[i] is not None:
                    var_map.update({self.indexed_vars[i][0]: str(args[i])})
                else:
                    raise IndexError
            except IndexError:
                var_map.update({self.indexed_vars[i][0]: self.indexed_vars[i][1]})
        return calc(self.expression, var_map)

def rpn_and_calc(tokens: list[str], var_map: dict[str, str]) -> decimal.Decimal | None | int:
    """
    Converts list of tokens to RPN and then calcs to value
    :param tokens: list of tokens
    :param var_map: dict of variables
    :return: value of RPN-converted tokens
    """
    operators = vars.OPERATORS
    ops = operators.keys()

    output: list[Union[decimal.Decimal, int]] = []
    stack_ops: list[str] = []  # list of operations
    stack_functions: list[tuple[str, list, list]] = []  #{function}, {tokens}, {args}

    def call_operator(operator: str):
        try:
            a, b = output[-2], output[-1]
        except IndexError:
            raise SyntaxError("Unfinished line")
        del output[-2:]

        op_to_run = operators[operator]
        if op_to_run[3]:
            for validator in op_to_run[3]:
                validator(a, b, op=operator)
        try:
            if a != 0 and b!=0:
                abs_a, abs_b = float(abs(a)), float(abs(b)) #type: ignore
                if operator == "**":
                    n_digits = float(b*decimal.Decimal(log10(abs_a)))
                elif operator == "*":
                    n_digits = log10(abs_b)+log10(abs_b)
                elif operator in ["/", "//"]:
                    n_digits = log10(abs_a) - log10(abs_b)

                elif operator == "+":
                    n_digits = max(log10(abs_a), log10(abs_b))
                else:
                    n_digits = -1
                n_digits = floor(n_digits)+1

                if n_digits > cst.MAXIMUM_DIGITS:
                    raise ValueError(f"Operation {a}**{b} will lead to at leat {n_digits} out of maximum of {cst.MAXIMUM_DIGITS}")

                if n_digits > cst.MAXIMUM_DIGITS_WARNING:
                    logger.warning(f"Operation {a} ** {b} will lead to at least {n_digits}(warning set on {cst.MAXIMUM_DIGITS_WARNING})")

            to_app = op_to_run[1](a, b)
            if check_is_integer(to_app):
                to_app = int(to_app) # type: ignore
            output.append(to_app)
        except TypeError:
            to_app = op_to_run[1](a, b, var_map = var_map)
            if check_is_integer(to_app):
                to_app = int(to_app) # type: ignore
            output.append(to_app)

        except decimal.InvalidOperation:
            raise TypeError(f"Cannot apply {operator} to '{a}' and '{b}'")

    for t in tokens:
        if t in ['', " "]:
            continue

        if t in vars.FUNCTIONS:
            stack_functions.append((t, [], []))
            continue

        elif len(stack_functions):
            last_func = stack_functions[-1]

            if t == "[":
                continue

            elif t == ',':
                last_func[2].append(rpn_and_calc(last_func[1], var_map))
                last_func[1].clear()

            elif t == "]":
                last_func[2].append(rpn_and_calc(last_func[1], var_map))

                args = last_func[2]
                func = vars.FUNCTIONS_CALLABLE_ENUM[last_func[0]][0]
                validators = vars.FUNCTIONS_CALLABLE_ENUM[last_func[0]][1]
                if validators:
                    for val in validators:
                        val(*args, op=last_func[0])
                if len(args) == 1:
                    logger.debug(f"Calling function {last_func[0]}({args[0]})")
                    try:
                        res = func(args[0], var_map = var_map)
                    except TypeError:
                        res = func(args[0])
                else:
                    logger.debug(f"Calling function {last_func[0]}({args})")
                    try:
                        res = func(*args, var_map = var_map)
                    except TypeError:
                        res = func(*args)
                stack_functions.pop()
                if len(stack_functions):
                    stack_functions[-1][1].append(str(res))
                else:
                    output.append(res)

            else:
                last_func[1].append(t)

        else:
            try:
                to_app = decimal.Decimal(t)
                if check_is_integer(to_app):
                    to_app = int(to_app) # type: ignore
                output.append(to_app)
                continue
            except decimal.InvalidOperation:

                if t in var_map.keys():
                    to_app = decimal.Decimal(calc(var_map[t], var_map))
                    if check_is_integer(to_app):
                        to_app = int(to_app) # type: ignore
                    output.append(to_app)
                elif t == "(":
                    stack_ops.append(t)
                elif t in ops:
                    if len(stack_ops):
                        if stack_ops[-1] == "(":
                            stack_ops.append(t)
                            continue
                        prev = stack_ops[-1]
                        prev_op = operators[prev]
                        cur_op = operators[t]
                        while (-2*(prev_op[2] is True and cur_op[2] is True)+1)*prev_op[0]>=cur_op[0]:
                            call_operator(stack_ops.pop())
                            if not len(stack_ops) or stack_ops[-1] == "(":
                                break
                    stack_ops.append(t)
                elif t == ")":
                    while stack_ops[-1] != "(":
                        op = stack_ops.pop()
                        if len(output) == 1:
                            raise SyntaxError("Unfinished line")
                        call_operator(op)
                        if not len(stack_ops) or stack_ops[-1] == "(":
                            break
                    stack_ops.pop()
                else:
                    raise InvalidTokenError(f"Unknown token: '{t}'", exc_type="unknown_token")

    for op in stack_ops[::-1]:
        call_operator(op)

    if len(output):
        return output[0]
    return None


def tokenize(compiled_expression: CompiledValidExpression, var_map: dict[str, str]) -> list[str]:
    """
    Tokenizes the expression
    :param compiled_expression -- raw mathematical expression needed to be tokenized(after compilation)
    :param var_map: map of variables: {"var_name": "var_expression"}
    :return: list of tokens
    """

    expression = compiled_expression.expression


    tokens: list[str] = [""]

    def place_token(token: str):
        if tokens[-1] == " ":
            tokens[-1] = token
        else:
            tokens.append(token)

    digitable = vars.FUNCTIONS+list(string.digits)
    if var_map:
        digitable+=list(var_map.keys())
    def check_is_digit(x: str) -> bool:
        if x in digitable or x.isnumeric():
            return True
        elif len(x)==0:
            return False
        elif x[-1] in [".", "e"] and x[:-1].isnumeric():
            return True
        else:
            try:
                float(x)
                return True
            except ValueError:
                return False

    known_tokens = digitable+list(vars.OPERATORS.keys())+["[", "]", "(", ")", ',', '.', 'e', ' ', '']
    current_functions: list[tuple[list[str], list[int], list[int]]] = []  #list of parsing functions. ({func_symbol}, {arg_count}, {parenthesis_count})
    for expr_ind in range(len(expression)):
        s = expression[expr_ind]
        if s == " ":
            tokens.append(" ")
            continue
        def filter_func(x: str):
            return x.startswith(tokens[-1]) or x.startswith(s)
        multi_symbols = list(filter(filter_func, vars.OPERATORS.keys()))
        multi_symbols+=list(filter(filter_func, vars.FUNCTIONS_CALLABLE_ENUM.keys()))
        if var_map:
            multi_symbols+=list(filter(filter_func, var_map.keys()))
        if any(tokens[-1]+s in ms for ms in multi_symbols) and len(multi_symbols):
            tokens[-1]+=s

        elif any(s in ms and s not in ['-', "+"] for ms in multi_symbols) and len(multi_symbols):
            place_token(s)
        else:
            if tokens[-1] in vars.FUNCTIONS:
                current_functions.append(([tokens[-1]], [0], [0]))

            elif tokens[-1] not in known_tokens and not check_is_digit(tokens[-1]):
                raise InvalidTokenError(f"Unknown token: '{tokens[-1]}'", exc_type="unknown_token")
            if tokens[-1] in digitable:
                for func in current_functions:
                    if func[1][0] == 0:
                        func[1][0] += 1
            if s.isdigit() or s in (".", "e"):  #if current symbol is digit
                if not check_is_digit(tokens[-1]):  #if previous one was not a digit we append the symbol as the first digit of a number
                    place_token(s)
                    if check_is_digit(tokens[-1]) and check_is_digit(tokens[-2]):
                        for digit in expression[expr_ind+1:]:
                            if check_is_digit(tokens[-1]+digit) or digit in [".", "e"]:
                                tokens[-1]+=digit
                            else:
                                break

                        raise SyntaxError(f"Two numbers without an operator: {tokens[-2]} {tokens[-1]}")
                else:
                    tokens[-1] += s  #if the previous one was a digit, we concatenate current token with symbol

                continue  #continues the loop(to eliminate extra checks)
            elif s == "-":
                if not check_is_digit(tokens[-1]):  #context management: if previous one was not a digit, it is a unary "-"
                    tokens.extend(("-1", "*"))
                    logger.warning(f"Two operations({tokens[-1]}-), '-' is unary one after other detected")
                else:
                    place_token("-")  #else it is a substraction
            elif s == "+" and not check_is_digit(tokens[-1]):
                #tokens.append("+")
                logger.warning(f"Two operations({tokens[-1]}+), '+' is unary one after other detected")
                continue

            elif not check_is_digit(tokens[-1]) and s in vars.OPERATORS.keys():

                for i in range(1, len(tokens)-1):
                    if check_is_digit(tokens[-i]):
                        break
                    elif tokens[-i] in vars.OPERATORS.keys():
                        raise SyntaxError(f"Two operations one after other: {tokens[-i]}{s}")
            elif s == ")":
                for i in range(1, len(tokens) - 1):
                    if check_is_digit(tokens[-i]):
                        break
                    elif tokens[-i] in vars.OPERATORS.keys():
                        raise SyntaxError(f"Operation '{tokens[-i]}' has no second operand")
                    elif tokens[-i] == "(":
                        if not len(current_functions):
                            raise InvalidParenthesisError("Empty parenthesis", exc_type="empty")
                        if current_functions[-1][2][0] != 0:
                            raise InvalidParenthesisError("Empty parenthesis", exc_type="empty")
                        else:
                            break

                tokens.append(")")
            else:
                place_token(s)#appending the other tokens
            if len(current_functions) > 0:
                cur_func = current_functions[-1]
                if s == "(":
                    if "[" not in cur_func[0][0]:
                        cur_func[0][0] += "["

                        tokens[-1] = "["
                    else:
                        current_functions[-1][2][0] += 1
                    continue
                elif s == ")":
                    cur_func[2][0] -= 1
                    if cur_func[2][0] < 0:
                        min_func_args = vars.FUNCTIONS_ARGS[cur_func[0][0][:-1]][0]
                        if min_func_args != -1 and min_func_args > cur_func[1][0]:
                            func_name = cur_func[0][0][:-1]
                            raise TypeError(f"TypeError: {func_name} requires minimum  of {min_func_args} arguments but {cur_func[1][0]} were given")

                        tokens[-1] = "]"

                        current_functions.pop()
                        continue

                elif s == ",":
                    cur_func[1][0] += 1
                    args: int = cur_func[1][0]
                    max_func_args = vars.FUNCTIONS_ARGS[cur_func[0][0][:-1]][1]
                    if max_func_args != -1 and max_func_args < args:
                        func_name = cur_func[0][0][:-1]

                        raise TypeError(f"TypeError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")

    logger.debug(f"{tokens=}")
    if tokens[-1] in vars.OPERATORS.keys():
        raise SyntaxError("Unfinished line")
    return tokens


def recover_state(resets: dict):
    vars.OPERATORS = resets["OPERATORS"]
    vars.FUNCTIONS_CALLABLE_ENUM = resets["FUNCTIONS_CALLABLE_ENUM"]
    vars.FUNCTIONS_ARGS = resets["FUNCTIONS_ARGS"]
    vars.FUNCTIONS = resets["FUNCTIONS"]
    logger.debug(f"Reset vars to initial state: {resets}")

@log_exceptions(logger = logger)
def calc(expr: str, var_map = None, debug_info = False, reset_on_exit = False) -> decimal.Decimal | int | None | tuple[Union[decimal.Decimal, None, int], Any, Any, Any]:
    """
    Calculates value of the expression
    :param expr: Expression to calculate
    :param var_map: map of variables {"var_name": "var_expression"}
    :param debug_info: if True, will return (result, vars.FUNCTIONS_CALLABLE_ENUM, FUNCTION_ARGS, OPERATORS)
    :param reset_on_exit: if True will reset OPERATORS and FUNCTIONS_CALLABLE_ENUM to initial state
    :return: (error code, value of the expression). Value can be returned as None if an error occurred during calculation process

    """
    global INITIAL_VALUES

    pre_compiled = PreCompiledValidExpression(expr)
    expression = CompiledExpression(pre_compiled.expression, var_map)


    if not var_map:
        var_map = expression.var_map
    func_map = expression.func_map
    for k, v in func_map.items():
        obj = CustomFunctionExecutor(v[0], v[1])
        vars.FUNCTIONS_CALLABLE_ENUM[k] = (obj.execute_function, None)
        vars.FUNCTIONS_ARGS[k] = (v[2], v[3])

    args = [('l', None), ('r', None)] #type: ignore
    for k, v in expression.op_map.items():
        obj = CustomFunctionExecutor(args, v[1]) #type: ignore
        pl = (v[0], obj.execute_function, v[2], None)
        logger.debug(f"setting new operator {k} as {pl}")
        vars.OPERATORS[k] = pl

    vars.FUNCTIONS = list(vars.FUNCTIONS_CALLABLE_ENUM.keys())
    compiled_expression = CompiledValidExpression(expression)


    logger.debug(f"{compiled_expression.expression=}")
    tokens = tokenize(compiled_expression, var_map)
    if not tokens:
        return None
    result = rpn_and_calc(tokens, var_map)
    if debug_info:
        return result, vars.FUNCTIONS_CALLABLE_ENUM.copy(), vars.FUNCTIONS.copy(), vars.OPERATORS.copy()

    if reset_on_exit:
        recover_state(INITIAL_VALUES)
    if not result:
        return result
    if type(result) is int:
        return result
    result = round_decimal(result) # type: ignore
    if check_is_integer(result):
        return int(result)
    else:
        return result


def main():
    """
    Entry point for application. Gets expression for stdin and passes it to calc() function
    """
    while True:

        expression: str = input("Enter the expression to calculate(q to exit): ")
        if expression == "q":
            exit(0)
        try:
            result = calc(expression, reset_on_exit=True)
        except Exception:
            result = None
        try:
            if cst.SCIENTIFIC_FORM:
                result = ("{:."+str(cst.SCIENTIFIC_FORM)+"E}").format(decimal.Decimal(result))
            logger.info(f"{expression} = {result}")
        except ValueError:
            logger.error(f"Could not calculate expression {expression}: too many digits")
            raise


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(cst.LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(stdout),
        ],
        format=cst.FORMAT
    )
    #logger.info(tokenize(CompiledExpression("max((2//3), 2)")))
    main()
