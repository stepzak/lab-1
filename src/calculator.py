import decimal
import logging
import sys
from math import floor
from typing import Union, Any

from compiler import CompiledExpression
from extra.exceptions import InvalidTokenError
from extra.types import Function, Operator, Variable, Context
from extra.utils import check_is_integer, log_exception
from tokenizer import Tokenizer
from validator import CompiledValidExpression, PreCompiledValidExpression
from vars import custom_log10
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

    def execute_function(self, *args, ctx: Context) -> decimal.Decimal:
        """
        Executes user defined function
        :param args: numbers passed as arguments to the function
        :param ctx: Context
        :return: function result
        """
        ctx.outer_names_buffer.append(self.name)
        for i in range(len(self.indexed_vars)):
            try:
                if args[i] is not None:
                    ctx.variables.update({self.indexed_vars[i][0]: Variable(args[i], True)})
                else:
                    raise IndexError
            except IndexError:
                ctx.variables.update({self.indexed_vars[i][0]: Variable(self.indexed_vars[i][1], True)}) #type: ignore
        new_calc = Calculator(ctx = ctx)

        return new_calc.calc(self.expression)


class Calculator:

    """
    Class for initializing and passing scopes while calculating(variables, functions, operators)
    :param ctx: Context
    """

    def __init__(self, ctx: Context = Context()):
        self.ctx = ctx
        self.tokens = ['  ']
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"ctx: {self.ctx}")

    @log_exception
    def calc(self, expression: str, tokens: list | None = None) -> decimal.Decimal | int | None | tuple[
        Union[decimal.Decimal, None, int], Any, Any, Any]:
        """
        Calculates value of the expression
        :return: value of expression. Value can be returned as None if an error occurred during calculation process
        """
        self.logger.debug(f"expression: {expression}")
        if not tokens:
            expression = str(expression)
            pre_compiled = PreCompiledValidExpression(' '.join(expression.split()))
            c_expression = CompiledExpression(pre_compiled.expression, self.ctx)

            if not self.ctx.variables:
                self.ctx.variables = c_expression.ctx.variables
            for k, v in c_expression.ctx.functions.items():
                if not isinstance(v, Function):
                    obj = CustomFunctionExecutor(k, v.indexed_args, v.expression)  #type: ignore
                    self.ctx.functions[k] = Function(obj.execute_function, [], v.min_args, v.max_args)
            args = [('l', None), ('r', None)]
            for op_name, op_data in c_expression.ctx.operators.items():
                if isinstance(op_data, Operator):
                    continue

                obj = CustomFunctionExecutor(op_name, args, op_data.expression)  # type: ignore
                pl = Operator(op_data.priority, obj.execute_function, op_data.is_right, [])
                self.logger.debug(f"setting new operator {op_name} as {pl}")
                self.ctx.operators[op_name] = pl

            compiled_expression = CompiledValidExpression(c_expression)

            self.logger.debug(f"{compiled_expression.expression=}")
            tokenizer = Tokenizer(ctx = self.ctx, logger = self.logger)
            self.tokens = tokenizer.tokenize(compiled_expression)
            if not self.tokens:
                return None
        else:
            self.tokens = tokens

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
        operators = self.ctx.operators
        ops = operators.keys()

        output: list[Union[decimal.Decimal, int]] = []
        stack_ops: list[str] = []  # list of operations
        stack_functions: list[tuple[str, list, list]] = []  # {function}, {tokens}, {args}

        func_map = self.ctx.functions
        funcs = func_map.keys()

        var_map = self.ctx.variables

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
                    abs_a, abs_b = abs(a), abs(b)  # type: ignore
                    if operator == "**":
                        n_digits: decimal.Decimal = b * custom_log10(abs_a)
                    elif operator == "*":
                        n_digits = custom_log10(abs_a) + custom_log10(abs_b)
                    elif operator in ["/", "//"]:
                        n_digits = custom_log10(abs_a) - custom_log10(abs_b)

                    elif operator == "+":
                        n_digits = max(custom_log10(abs_a), custom_log10(abs_b))
                    else:
                        n_digits = decimal.Decimal(-1)
                    n_digits = floor(n_digits) + 1 #type: ignore

                    if n_digits > cst.MAXIMUM_DIGITS:
                        raise ValueError(
                            f"Operation {a} {operator} {b} will lead to at leat {n_digits} out of maximum of {cst.MAXIMUM_DIGITS}")

                    if n_digits > cst.MAXIMUM_DIGITS_WARNING:
                        self.logger.warning(
                            f"Operation {a} {operator} {b} will lead to at least {n_digits}(warning set on {cst.MAXIMUM_DIGITS_WARNING})")

                to_app = decimal.Decimal(op_to_run.callable_function(a, b))
                if check_is_integer(to_app):
                    to_app = int(to_app)  # type: ignore
                output.append(to_app)
            except TypeError:
                to_app = op_to_run.callable_function(a, b, ctx = self.ctx)
                to_app = decimal.Decimal(to_app)
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
                    new_calc = Calculator(self.ctx)
                    last_func[2].append(new_calc.calc(expression = "", tokens = last_func[1]))
                    last_func[1].clear()

                elif t == "]":
                    new_calc = Calculator(self.ctx)
                    last_func[2].append(new_calc.calc(expression = "", tokens = last_func[1]))

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
                            res = func(args[0], ctx = self.ctx)
                        except TypeError:
                            res = func(args[0])
                    else:
                        self.logger.debug(f"Calling function {last_func[0]}({args})")
                        try:
                            res = func(*args, ctx = self.ctx)
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

                    if t in var_map.keys():
                        var = var_map[t]
                        if not var.local:
                            self.ctx.outer_names_buffer.append(t)
                        cls = Calculator(self.ctx)
                        result = cls.calc(var.value)
                        if not var.local:
                            self.ctx.outer_names_buffer.pop()
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

        if len(output) >= 1:
            if len(output) == 1:
                return output[0]
            raise SyntaxError(f"Unfinished line, some numbers left without operations: {output}")
        return None
