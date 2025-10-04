import decimal
import logging
import string
import sys
from math import log10, floor
from typing import Union, Any

from compiler import CompiledExpression
from extra.exceptions import InvalidParenthesisError, InvalidTokenError
from extra.types import Function, Operator, Variable
from extra.utils import check_is_integer, log_exception
from validator import CompiledValidExpression, PreCompiledValidExpression
from vars import FUNCTIONS_CALLABLE_ENUM, OPERATORS
import constants as cst
from decimal import getcontext

getcontext().prec = cst.PRECISION

sys.set_int_max_str_digits(cst.MAXIMUM_DIGITS)

class CustomFunctionExecutor:
    """
    Class to call user defined functions
    :param indexed_args: list of tuples: (arg_name, arg_default_value | None)
    :param: expression: expression defined by user after return statement
    """
    def __init__(self, name: str, indexed_args: list[tuple[str, Union[str, None]]],  expression: str):
        self.name = name
        self.expression = expression
        self.indexed_vars = indexed_args

    def execute_function(self, *args, var_map: dict, func_map: dict[str, Function],
                         op_map: dict[str, Operator], outer_names: list[str]) -> decimal.Decimal:
        """
        Executes user defined function
        :param args: numbers passed as arguments to the function
        :param var_map: map from variable name to variable expression
        :param func_map: map from defined function name to its Function type dataclass
        :param op_map: map from defined operator to its Operator type dataclass
        :param outer_names: list of names, calculations for which were called recursively(variables, functions)
        :return: function result
        """
        outer_names = outer_names or []
        outer_names.append(self.name)
        var_map = var_map.copy()
        for i in range(len(self.indexed_vars)):
            try:
                if args[i] is not None:
                    var_map.update({self.indexed_vars[i][0]: Variable(args[i], True)})
                else:
                    raise IndexError
            except IndexError:
                var_map.update({self.indexed_vars[i][0]: Variable(self.indexed_vars[i][1], True)}) #type: ignore
        new_calc = Calculator(self.expression, var_map, func_map, op_map, outer_names_buffer=outer_names)

        return new_calc.calc()


class Calculator:

    """
    Class for initializing and passing scopes while calculating(variables, functions, operators)
    :param expression: expression to calculate
    :param var_map: map from variable name to its Variable dataclass
    :param func_map: map from defined function name to its Function type dataclass
    :param op_map: map from defined operator to its Operator type dataclass
    :param outer_names_buffer: list of names, calculations for which were called recursively(variables, functions)
    """

    def __init__(self, expression: str, var_map = None, func_map = None, op_map = None, outer_names_buffer = None):
        self.expression = expression
        self.var_map: dict[str, Variable] = var_map or {}
        self.func_map: dict[str, Function] = FUNCTIONS_CALLABLE_ENUM.copy()
        self.func_map.update(func_map or {})
        self.op_map: dict[str, Operator] = OPERATORS.copy()
        self.op_map.update(op_map or {})
        self.tokens = ['  ']
        self.logger = logging.getLogger(__name__)
        self.outer_names_buffer = outer_names_buffer or []
        self.logger.debug(f"outer_names_buffer: {self.outer_names_buffer}")

    @log_exception
    def calc(self) -> decimal.Decimal | int | None | tuple[
        Union[decimal.Decimal, None, int], Any, Any, Any]:
        """
        Calculates value of the expression
        :return: value of expression. Value can be returned as None if an error occurred during calculation process
        """

        pre_compiled = PreCompiledValidExpression(self.expression)
        expression = CompiledExpression(pre_compiled.expression, self.var_map, self.func_map, self.op_map)

        if not self.var_map:
            self.var_map = expression.var_map
        for k, v in expression.func_map.items():
            if not isinstance(v, Function):
                obj = CustomFunctionExecutor(k, v[0], v[1])  #type: ignore
                self.func_map[k] = Function(obj.execute_function, [], v[2], v[3])
        args = [('l', None), ('r', None)]
        for k, v in expression.op_map.items():
            if isinstance(v, Operator):
                continue
            obj = CustomFunctionExecutor(k, args, v[1])  # type: ignore
            pl = Operator(v[0], obj.execute_function, v[2], [])
            self.logger.debug(f"setting new operator {k} as {pl}")
            self.op_map[k] = pl

        compiled_expression = CompiledValidExpression(expression)

        self.logger.debug(f"{compiled_expression.expression=}")
        self.tokens = self.tokenize(compiled_expression)
        if not self.tokens:
            return None

        result = self.rpn_and_calc()

        if type(result) is int:
            return result
        return result

    @log_exception
    def rpn_and_calc(self) -> decimal.Decimal | None | int:
        """
        Converts list of tokens to RPN and then calcs their value
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

            for validator in op_to_run.validators:
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

                to_app = op_to_run.callable_function(a, b)
                if check_is_integer(to_app):
                    to_app = int(to_app)  # type: ignore
                output.append(to_app)
            except TypeError:
                to_app = op_to_run.callable_function(a, b, var_map=self.var_map, func_map=func_map, op_map=self.op_map, outer_names = self.outer_names_buffer)
                if check_is_integer(to_app):
                    to_app = int(to_app)  # type: ignore
                output.append(to_app)

            except decimal.InvalidOperation:
                raise TypeError(f"Cannot apply {operator} to '{a}' and '{b}'")

        for t in self.tokens:
            if t.isspace():
                continue

            if t in funcs:
                stack_functions.append((t, [], []))
                continue

            elif len(stack_functions):
                last_func = stack_functions[-1]

                if t == "[":
                    continue

                elif t == ',':
                    new_calc = Calculator("", self.var_map, func_map, self.op_map, self.outer_names_buffer)
                    new_calc.tokens = last_func[1]
                    last_func[2].append(new_calc.rpn_and_calc())
                    last_func[1].clear()

                elif t == "]":
                    new_calc = Calculator("", self.var_map, func_map, self.op_map, self.outer_names_buffer)
                    new_calc.tokens = last_func[1]
                    last_func[2].append(new_calc.rpn_and_calc())

                    args = last_func[2]
                    func = func_map[last_func[0]].callable_function
                    validators = func_map[last_func[0]].validators

                    for val in validators:
                        try:
                            val(*args, op=last_func[0])
                        except Exception:
                            raise
                    if len(args) == 1:
                        self.logger.debug(f"Calling function {last_func[0]}({args[0]})")
                        try:
                            res = func(args[0], var_map=self.var_map, func_map=func_map, op_map=self.op_map, outer_names  = self.outer_names_buffer)
                        except TypeError:
                            res = func(args[0])
                    else:
                        self.logger.debug(f"Calling function {last_func[0]}({args})")
                        try:
                            res = func(*args, var_map=self.var_map, func_map=func_map, op_map=self.op_map, outer_names = self.outer_names_buffer)
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
                        var = self.var_map[t]
                        if not var.local:
                            self.outer_names_buffer.append(t)
                        cls = Calculator(self.var_map[t].value, self.var_map, func_map, self.op_map, self.outer_names_buffer)
                        result = cls.calc()
                        if not var.local:
                            self.outer_names_buffer.pop()
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
                            while (-2 * (prev_op.is_right is True and cur_op.is_right is True) + 1) * prev_op.priority >= cur_op.priority:
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
        :param compiled_expression: raw mathematical expression needed to be tokenized(after compilation and scope defining)
        :return: list of tokens
        """

        expression = compiled_expression.expression.strip()

        tokens: list[str] = ["  "]

        def place_token(token: str):
            """
            For extra conditions in the future
            """
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

        known_tokens = digitable + list(self.op_map.keys()) + ["[", "]", "(", ")", ',', '.', 'e', ' ', '', "  "]
        current_functions: list[tuple[list[str], list[int], list[
            int]]] = []  # list of parsing functions. ({func_symbol}, {arg_count}, {parenthesis_count})
        for expr_ind in range(len(expression)):
            s = expression[expr_ind]
            if len(tokens)>=2:
                rec_error = False
                if tokens[-2] not in self.op_map.keys():
                    rec_error = tokens[-2] in self.outer_names_buffer
                elif len(self.outer_names_buffer):
                    rec_error = tokens[-2] == self.outer_names_buffer[-1]

                if rec_error:
                    raise RecursionError(f"Name '{tokens[-2]}' defined with itself({expression}). Recursion is not (yet) supported")
                if len(tokens)>2:
                    if tokens[-2].isspace() and check_is_digit(tokens[-1])==check_is_digit(tokens[-3]):
                        while check_is_digit(tokens[-1]+s):
                            tokens[-1]+=s
                            expr_ind+=1
                            if len(expression)-1<expr_ind:
                                break
                            s = expression[expr_ind]
                        raise SyntaxError(f"Missed operation between {tokens[-3]} and {tokens[-1]}")
            if s == " " and not tokens[-1].isspace():
                tokens.append(" ")
                continue

            def filter_func(x: str):
                return x.startswith(tokens[-1]) or x.startswith(s)

            multi_symbols = list(filter(filter_func, self.op_map.keys()))
            multi_symbols += list(filter(filter_func, self.func_map.keys()))
            if self.var_map:
                multi_symbols += list(filter(filter_func, self.var_map.keys()))

            extra_check_for_unary = True and len(tokens)>1
            if s in "+-" and len(tokens)>1:
                try:
                    extra_check_for_unary = not check_is_digit(expression[expr_ind + 1])
                except IndexError:
                    raise InvalidTokenError(f"Unfinished line: operation '{s}' has no second operand",
                                            exc_type="invalid_token")

            if any(tokens[-1] + s in ms for ms in multi_symbols) and len(multi_symbols):
                tokens[-1] += s

            elif any(s in ms and s for ms in multi_symbols) and len(multi_symbols) and extra_check_for_unary:
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
                    if not check_is_digit(tokens[-1]):  # if previous one was not a digit we append the symbol as the first digit of a number
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
                        if res not in "()[] " and not res.isspace():
                            self.logger.warning(f"Two operations({res}-), '-' is unary: one after other detected")
                    else:
                        place_token("-")  # else it is a substraction
                elif s == "+":
                    for i in range(1, len(tokens)+1):
                        if not tokens[-i].isspace():
                            res = tokens[-i]
                            break
                    else:
                        res = tokens[-1]
                    if not check_is_digit(res) and res not in "()[]" and not res.isspace():
                        self.logger.warning(f"Two operations({res}+), '+' is unary: one after other detected")
                    elif check_is_digit(res) or res in "()[]":
                        tokens.append("+")
                    continue

                elif not check_is_digit(tokens[-1]) and s in self.op_map.keys():

                    for i in range(1, len(tokens) - 1):
                        if check_is_digit(tokens[-i]):
                            break
                        elif tokens[-i] in self.op_map.keys():
                            raise SyntaxError(f"Two operations one after other: {tokens[-i]}{s}")
                    place_token(s)
                elif s == ")":
                    for i in range(1, len(tokens) - 1):
                        if check_is_digit(tokens[-i]):
                            break
                        elif tokens[-i] in self.op_map.keys():
                            raise SyntaxError(f"Unfinished line: operation '{tokens[-i]}' has no second operand")
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
                            min_func_args = self.func_map[cur_func[0][0][:-1]].min_args
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
                        max_func_args = self.func_map[cur_func[0][0][:-1]].max_args
                        if max_func_args != -1 and max_func_args < args:
                            func_name = cur_func[0][0][:-1]

                            raise TypeError(
                                f"TypeError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")

        self.logger.debug(f"{tokens=}")
        for ind in range(1, 3):
            t = tokens[-ind]
            if t in self.outer_names_buffer:
                raise RecursionError(f"Name '{t}' defined with itself({expression}). Recursion is not (yet) supported")
        for t in tokens[::-1]:
            if t in self.op_map.keys() and t not in "+-":
                raise InvalidTokenError(f"Unfinished line: operation '{t}' has no second operand",
                                        exc_type="invalid_token")
            elif check_is_digit(t):
                break

        for t in tokens:
            if t in self.op_map.keys() and t not in "+-":
                raise InvalidTokenError(f"Unfinished line: operation '{t}' has no first operand",
                                        exc_type="invalid_token")
            elif check_is_digit(t):
                break

        return tokens
