import copy
import decimal
import logging
import sys
from math import floor
from typing import Union

from src.compiler import CompiledExpression
from src.extra.exceptions import InvalidTokenError
from src.extra.types import Function, Operator, Variable
from src.extra.context_type import Context
from src.extra.utils import check_is_integer, log_exception
from src.tokenizer import Tokenizer
from src.validator import CompiledValidExpression, PreCompiledValidExpression
from src.vars import custom_log10
import src.constants as cst
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
        ctx = copy.deepcopy(ctx)
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

    def __init__(self, ctx: Context | None = None):
        self.ctx = copy.deepcopy(ctx) or Context()
        self.tokens = ['  ']
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"ctx: {self.ctx}")

    @log_exception
    def calc(self, expression: str, tokens: list | None = None) -> decimal.Decimal | int | None:
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
                if not getattr(v, "callable_function", None):
                    obj = CustomFunctionExecutor(k, v.indexed_args, v.expression)  #type: ignore
                    self.ctx.functions[k] = Function(obj.execute_function, [], v.min_args, v.max_args)
            args = [('l', None), ('r', None)]
            for op_name, op_data in c_expression.ctx.operators.items():
                if getattr(op_data, "callable_function", None):
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

        rpn = self.rpn()
        result = self.calc_rpn(rpn)
        if type(result) is int:
            return result
        result = decimal.Decimal(result)
        return result

    def calc_rpn(self, rpn: list[str | int | list]): # [1, 2, '+', 'max', [1, 3, '+', ',', 'min', [1, ',', 3] ] ]
        operators = self.ctx.operators
        ops = operators.keys()
        func_map = self.ctx.functions
        funcs = func_map.keys()

        var_map = self.ctx.variables

        def call_current_operator():
            try:
                operator = rpn.pop(i)
                a = rpn.pop(i-1)
                b = rpn.pop(i-2)
            except IndexError:
                raise SyntaxError("Unfinished line")

            op_to_run: Operator = operators[operator] #type: ignore

            for validator in op_to_run.validators:
                validator(a, b, op=operator)

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
            try:

                to_app = decimal.Decimal(op_to_run.callable_function(a, b))
            except TypeError:
                to_app = op_to_run.callable_function(a, b, ctx = self.ctx)
                to_app = decimal.Decimal(to_app)
            except decimal.InvalidOperation:
                raise TypeError(f"Cannot apply {operator} to '{a}' and '{b}'")
            if check_is_integer(to_app):
                to_app = int(to_app)  # type: ignore
            rpn.insert(i, to_app)


        i = -1
        while i < len(rpn)-1:
            i+=1
            t = rpn[i]
            if t in var_map.keys():
                v = var_map[t] #type: ignore
                new_calc = Calculator(self.ctx)
                var_val = new_calc.calc(v.value)
                rpn[i] = var_val

            elif t in ops:
                call_current_operator()

            elif t in funcs:
                current_tokens = [] #type: ignore
                current_args = []
                for arg in rpn[i+1]: #type: ignore
                    if arg == ',':
                        new_calc = Calculator(self.ctx)
                        arg_val = new_calc.calc_rpn(current_tokens)
                        current_args.append(arg_val)
                        current_tokens.clear()
                    else:
                        current_tokens.append(arg)

                new_calc = Calculator(self.ctx)
                arg_val = new_calc.calc_rpn(copy.deepcopy(current_tokens))
                current_args.append(arg_val)
                current_tokens.clear()
                func = func_map[t] #type: ignore
                try:
                    res = func.callable_function(*current_args, ctx = self.ctx)
                except TypeError:
                    res = func.callable_function(*current_args)

                rpn.insert(i, res)
                del rpn[i+1:i + 3]

        call_current_operator()
        return rpn[0]




    def rpn(self, tokens: list | None = None) -> list:
        """
        Converts list of tokens to RPN and then calcs their value
        :return: value of RPN-converted tokens
        """
        if not tokens:
            tokens = self.tokens
        operators = self.ctx.operators
        ops = operators.keys()
        func_map = self.ctx.functions
        funcs = func_map.keys()

        var_map = self.ctx.variables
        output = []
        stack_ops: list[str] = []  # list of operations
        stack_functions: list[tuple[str, list, list]] = []  # {function}, {tokens}, {args}


        for t in tokens:
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
                    last_func[2].extend(self.rpn(last_func[1])+[","])
                    last_func[1].clear()

                elif t == "]":

                    last_func[2].extend(self.rpn(last_func[1]))

                    stack_functions.pop()
                    if len(stack_functions):
                        stack_functions[-1][1].extend([last_func[0], last_func[2]])
                    else:
                        output.extend([last_func[0], last_func[2] ])

                else:
                    last_func[1].append(t)
            else:
                try:
                    to_app = decimal.Decimal(t)
                    if check_is_integer(to_app):
                        to_app = int(to_app)  # type: ignore
                    output.append(to_app) #type: ignore
                    continue
                except decimal.InvalidOperation:

                    if t in var_map.keys():
                        output.append(t)
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
                                output.append(stack_ops.pop())
                                if not len(stack_ops) or stack_ops[-1] == "(":
                                    break
                        stack_ops.append(t)
                    elif t == ")":
                        while stack_ops[-1] != "(":
                            op = stack_ops.pop()
                            if len(output) == 1:
                                raise SyntaxError("Unfinished line")
                            output.append(op)
                            if not len(stack_ops) or stack_ops[-1] == "(":
                                break
                        stack_ops.pop()
                    else:
                        raise InvalidTokenError(f"Unknown token: '{t}'", exc_type="unknown_token")

        for op in stack_ops[::-1]:
            output.append(op)

        return output
