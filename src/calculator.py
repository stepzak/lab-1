import decimal
import logging
import string
import sys
from math import log10, floor
from typing import Union, Any, Sequence

from compiler import CompiledExpression
from extra.exceptions import InvalidParenthesisError, InvalidTokenError
from extra.utils import check_is_integer, log_exception, round_decimal
from validator import CompiledValidExpression, PreCompiledValidExpression
from vars import FUNCTIONS_CALLABLE_ENUM, OPERATORS
import constants as cst
from decimal import getcontext

getcontext().prec = cst.PRECISION

sys.set_int_max_str_digits(cst.MAXIMUM_DIGITS)

class CustomFunctionExecutor:
    def __init__(self, indexed_vars: list[tuple[str, Union[str, None]]],  expression: str):
        self.expression = expression
        self.indexed_vars = indexed_vars

    def execute_function(self, *args, var_map: dict, func_map: dict, op_map: dict) -> decimal.Decimal:
        var_map = var_map.copy()
        for i in range(len(self.indexed_vars)):
            try:
                if args[i] is not None:
                    var_map.update({self.indexed_vars[i][0]: str(args[i])})
                else:
                    raise IndexError
            except IndexError:
                var_map.update({self.indexed_vars[i][0]: self.indexed_vars[i][1]})
        new_calc = Calculator(self.expression, var_map, func_map, op_map)
        return new_calc.calc()


class Calculator:


    def __init__(self, expression: str, var_map = None, func_map = None, op_map = None):
        self.expression = expression
        self.var_map = var_map
        self.func_map = FUNCTIONS_CALLABLE_ENUM.copy()
        self.func_map.update(func_map or {})
        self.op_map = OPERATORS.copy()
        self.op_map.update(op_map or {})
        self.tokens = ['']
        self.logger = logging.getLogger(__name__)



    @log_exception
    def calc(self) -> decimal.Decimal | int | None | tuple[
        Union[decimal.Decimal, None, int], Any, Any, Any]:
        """
        Calculates value of the expression
        :return: (error code, value of the expression). Value can be returned as None if an error occurred during calculation process

        """

        pre_compiled = PreCompiledValidExpression(self.expression)
        expression = CompiledExpression(pre_compiled.expression, self.var_map, self.func_map, self.op_map)

        if not self.var_map:
            self.var_map = expression.var_map
        #self.func_map = expression.func_map
        for k, v in expression.func_map.items():
            if isinstance(v[0], Sequence):
                obj = CustomFunctionExecutor(v[0], v[1]) #type: ignore
                self.func_map[k] = (obj.execute_function, None, v[2])
        args = [('l', None), ('r', None)]
        for k, v in expression.op_map.items():
              # type: ignore
            if callable(v[1]):
                continue
            obj = CustomFunctionExecutor(args, v[1])  # type: ignore
            pl = (v[0], obj.execute_function, v[2], None)
            self.logger.debug(f"setting new operator {k} as {pl}")
            self.op_map[k] = pl

        compiled_expression = CompiledValidExpression(expression)

        self.logger.debug(f"{compiled_expression.expression=}")
        self.tokens = self.tokenize(compiled_expression)
        if not self.tokens:
            return None

        result = self.rpn_and_calc()

        if not result:
            return result
        if type(result) is int:
            return result
        result = round_decimal(result)  # type: ignore
        if check_is_integer(result):
            return int(result)
        else:
            return result

    @log_exception
    def rpn_and_calc(self) -> decimal.Decimal | None | int:
        """
        Converts list of tokens to RPN and then calcs to value
        :return: value of RPN-converted tokens
        """
        operators = self.op_map
        ops = operators.keys()

        output: list[Union[decimal.Decimal, int]] = []
        stack_ops: list[str] = []  # list of operations
        stack_functions: list[tuple[str, list, list]] = []  # {function}, {tokens}, {args}

        func_map = self.func_map
        funcs = func_map.keys()

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
                if a != 0 and b != 0:
                    abs_a, abs_b = float(abs(a)), float(abs(b))  # type: ignore
                    if operator == "**":
                        n_digits = float(b * decimal.Decimal(log10(abs_a)))
                    elif operator == "*":
                        n_digits = log10(abs_b) + log10(abs_b)
                    elif operator in ["/", "//"]:
                        n_digits = log10(abs_a) - log10(abs_b)

                    elif operator == "+":
                        n_digits = max(log10(abs_a), log10(abs_b))
                    else:
                        n_digits = -1
                    n_digits = floor(n_digits) + 1

                    if n_digits > cst.MAXIMUM_DIGITS:
                        raise ValueError(
                            f"Operation {a}**{b} will lead to at leat {n_digits} out of maximum of {cst.MAXIMUM_DIGITS}")

                    if n_digits > cst.MAXIMUM_DIGITS_WARNING:
                        self.logger.warning(
                            f"Operation {a} ** {b} will lead to at least {n_digits}(warning set on {cst.MAXIMUM_DIGITS_WARNING})")

                to_app = op_to_run[1](a, b)
                if check_is_integer(to_app):
                    to_app = int(to_app)  # type: ignore
                output.append(to_app)
            except TypeError:
                to_app = op_to_run[1](a, b, var_map=self.var_map, func_map=func_map, op_map=self.op_map)
                if check_is_integer(to_app):
                    to_app = int(to_app)  # type: ignore
                output.append(to_app)

            except decimal.InvalidOperation:
                raise TypeError(f"Cannot apply {operator} to '{a}' and '{b}'")

        for t in self.tokens:
            if t in ['', " "]:
                continue

            if t in funcs:
                stack_functions.append((t, [], []))
                continue

            elif len(stack_functions):
                last_func = stack_functions[-1]

                if t == "[":
                    continue

                elif t == ',':
                    new_calc = Calculator("", self.var_map, func_map, self.op_map)
                    new_calc.tokens = last_func[1]
                    last_func[2].append(new_calc.rpn_and_calc())
                    last_func[1].clear()

                elif t == "]":
                    new_calc = Calculator("", self.var_map, func_map, self.op_map)
                    new_calc.tokens = last_func[1]
                    last_func[2].append(new_calc.rpn_and_calc())

                    args = last_func[2]
                    func = func_map[last_func[0]][0]
                    validators = func_map[last_func[0]][1]
                    if validators:
                        for val in validators:
                            try:
                                val(*args, op=last_func[0])
                            except Exception:
                                raise
                    if len(args) == 1:
                        self.logger.debug(f"Calling function {last_func[0]}({args[0]})")
                        try:
                            res = func(args[0], var_map=self.var_map, func_map=func_map, op_map=self.op_map)
                        except TypeError:
                            res = func(args[0])
                    else:
                        self.logger.debug(f"Calling function {last_func[0]}({args})")
                        try:
                            res = func(*args, var_map=self.var_map, func_map=func_map, op_map=self.op_map)
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
                        to_app = int(to_app)  # type: ignore
                    output.append(to_app)
                    continue
                except decimal.InvalidOperation:

                    if t in self.var_map.keys():
                        cls = Calculator(self.var_map[t], self.var_map, func_map, self.op_map)
                        result = cls.calc()
                        to_app = decimal.Decimal(result)
                        if check_is_integer(to_app):
                            to_app = int(to_app)  # type: ignore
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
                            while (-2 * (prev_op[2] is True and cur_op[2] is True) + 1) * prev_op[0] >= cur_op[0]:
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

    def tokenize(self, compiled_expression: CompiledValidExpression) -> list[str]:
        """
        Tokenizes the expression
        :param compiled_expression -- raw mathematical expression needed to be tokenized(after compilation)
        :return: list of tokens
        """

        expression = compiled_expression.expression.strip()

        tokens: list[str] = [""]

        def place_token(token: str):
            if tokens[-1] == " ":
                tokens[-1] = token
            else:
                tokens.append(token)

        digitable = list(self.func_map.keys()) + list(string.digits) + list(self.var_map.keys())

        def check_is_digit(x: str) -> bool:
            if x in digitable or x.isnumeric():
                return True
            elif len(x) == 0:
                return False
            elif x[-1] in [".", "e"] and x[:-1].isnumeric():
                return True
            else:
                try:
                    float(x)
                    return True
                except ValueError:
                    return False

        known_tokens = digitable + list(self.op_map.keys()) + ["[", "]", "(", ")", ',', '.', 'e', ' ', '']
        current_functions: list[tuple[list[str], list[int], list[
            int]]] = []  # list of parsing functions. ({func_symbol}, {arg_count}, {parenthesis_count})
        for expr_ind in range(len(expression)):

            s = expression[expr_ind]
            if s == " ":
                tokens.append(" ")
                continue

            def filter_func(x: str):
                return x.startswith(tokens[-1]) or x.startswith(s)

            multi_symbols = list(filter(filter_func, self.op_map.keys()))
            multi_symbols += list(filter(filter_func, self.func_map.keys()))
            if self.var_map:
                multi_symbols += list(filter(filter_func, self.var_map.keys()))
            if any(tokens[-1] + s in ms for ms in multi_symbols) and len(multi_symbols) and s not in "+-":
                tokens[-1] += s

            elif any(s in ms and s for ms in multi_symbols) and len(multi_symbols) and s not in "-+":
                    place_token(s)
            else:
                if tokens[-1] in self.func_map.keys():
                    current_functions.append(([tokens[-1]], [0], [0]))

                elif tokens[-1] not in known_tokens and not check_is_digit(tokens[-1]):
                    raise InvalidTokenError(f"Unknown token: '{tokens[-1]}'", exc_type="unknown_token")
                if tokens[-1] in digitable:
                    for func in current_functions:
                        if func[1][0] == 0:
                            func[1][0] += 1
                if s.isdigit() or s in (".", "e"):  # if current symbol is digit
                    if not check_is_digit(tokens[
                                              -1]):  # if previous one was not a digit we append the symbol as the first digit of a number
                        place_token(s)
                        if check_is_digit(tokens[-1]) and check_is_digit(tokens[-2]):
                            for digit in expression[expr_ind + 1:]:
                                if check_is_digit(tokens[-1] + digit) or digit in [".", "e"]:
                                    tokens[-1] += digit
                                else:
                                    break

                            raise SyntaxError(f"Two numbers without an operator: {tokens[-2]} {tokens[-1]}")
                    else:
                        tokens[-1] += s  # if the previous one was a digit, we concatenate current token with symbol

                    continue  # continues the loop(to eliminate extra checks)
                elif s == "-":
                    for i in range(1, len(tokens)+1):
                        if tokens[-i] != "" and tokens[-i] != ' ':
                            res = tokens[-i]
                            break
                    else:
                        res = tokens[-1]
                    if not check_is_digit(res):  # context management: if previous one was not a digit, it is a unary "-"
                        tokens.extend(("-1", "*"))
                        if res not in "()[] " and res!="":
                            self.logger.warning(f"Two operations({res}-), '-' is unary one after other detected")
                    else:
                        place_token("-")  # else it is a substraction
                elif s == "+":
                    # tokens.append("+")
                    for i in range(1, len(tokens)+1):
                        if tokens[-i] != "":
                            res = tokens[-i]
                            break
                    else:
                        res = tokens[-1]
                    if not check_is_digit(res) and res not in "()[]" and res!="":
                        self.logger.warning(f"Two operations({tokens[-1]}+), '+' is unary one after other detected")
                    else:
                        tokens.append("+")
                    continue

                elif not check_is_digit(tokens[-1]) and s in self.op_map.keys():

                    for i in range(1, len(tokens) - 1):
                        if check_is_digit(tokens[-i]):
                            break
                        elif tokens[-i] in self.op_map.keys():
                            raise SyntaxError(f"Two operations one after other: {tokens[-i]}{s}")
                elif s == ")":
                    for i in range(1, len(tokens) - 1):
                        if check_is_digit(tokens[-i]):
                            break
                        elif tokens[-i] in self.op_map.keys():
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
                    place_token(s)  # appending the other tokens
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
                            min_func_args = self.func_map[cur_func[0][0][:-1]][2][0]
                            if min_func_args != -1 and min_func_args > cur_func[1][0]:
                                func_name = cur_func[0][0][:-1]
                                raise TypeError(
                                    f"TypeError: {func_name} requires minimum  of {min_func_args} arguments but {cur_func[1][0]} were given")

                            tokens[-1] = "]"

                            current_functions.pop()
                            continue

                    elif s == ",":
                        cur_func[1][0] += 1
                        args: int = cur_func[1][0]
                        max_func_args = self.func_map[cur_func[0][0][:-1]][2][1]
                        if max_func_args != -1 and max_func_args < args:
                            func_name = cur_func[0][0][:-1]

                            raise TypeError(
                                f"TypeError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")

        self.logger.debug(f"{tokens=}")
        if tokens[-1] in OPERATORS.keys():
            raise SyntaxError("Unfinished line")
        return tokens
