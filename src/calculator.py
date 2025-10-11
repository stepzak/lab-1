import copy
import decimal
import logging
import sys
from math import floor
from typing import Union, Any
from src.compiler import CompiledExpression
from src.rpn import ConverterRPN
import src.extra.types as types
from src.extra.types import Context
import src.extra.utils as utils
from src.tokenizer import Tokenizer
import src.validator as validator
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
                    ctx.variables.update({self.indexed_vars[i][0]: types.Variable(args[i], True)})
                else:
                    raise IndexError
            except IndexError:
                ctx.variables.update({self.indexed_vars[i][0]: types.Variable(self.indexed_vars[i][1], True)}) #type: ignore
        new_calc = Calculator(ctx = ctx)

        return new_calc.calc(self.expression)

@utils.init_default_ctx
class Calculator:

    """
    Class for initializing and passing scopes while calculating(variables, functions, operators)
    :param ctx: Context
    """
    def __init__(self, rpn: list | None = None, *, ctx: Context | None = None):
        self.ctx: Context = ctx or Context({}, {}, [], {}, {})
        self.tokens = ['  ']
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"ctx: {self.ctx}")
        self.rpns = rpn or []

    @utils.log_exception
    def calc(self, expression: str, tokens: list | None = None) -> decimal.Decimal | int | None | tuple[
        Union[decimal.Decimal, None, int], Any, Any, Any]:
        """
        Calculates value of the expression
        :return: value of expression. Value can be returned as None if an error occurred during calculation process
        """
        self.logger.debug(f"expression: {expression}")
        if not tokens:
            expression = str(expression)
            pre_compiled = validator.PreCompiledValidExpression.create(' '.join(expression.split()))
            c_expression = CompiledExpression(pre_compiled.expression, ctx = self.ctx)

            if not self.ctx.variables:
                self.ctx.variables = c_expression.ctx.variables
            for k, v in c_expression.ctx.functions.items(): #type: ignore
                if not getattr(v, "callable_function", None):
                    obj = CustomFunctionExecutor(k, v.indexed_args, v.expression)  #type: ignore
                    self.ctx.functions[k] = types.Function(obj.execute_function, [], v.min_args, v.max_args) #type: ignore
            args = [('l', None), ('r', None)]
            for op_name, op_data in c_expression.ctx.operators.items():
                if getattr(op_data, "callable_function", None):
                    continue

                obj = CustomFunctionExecutor(op_name, args, op_data.expression)  # type: ignore
                pl = types.Operator(op_data.priority, obj.execute_function, op_data.is_right, [])
                self.logger.debug(f"setting new operator {op_name} as {pl}")
                self.ctx.operators[op_name] = pl #type: ignore

            compiled_expression = validator.CompiledValidExpression(c_expression)

            self.logger.debug(f"{compiled_expression.expression=}")
            tokenizer = Tokenizer(ctx = self.ctx, logger = self.logger) #type: ignore
            self.tokens = tokenizer.tokenize(compiled_expression)
            if not self.tokens:
                return None
        else:
            self.tokens = tokens

        self.rpns = ConverterRPN(self.ctx).rpn(self.tokens) #type: ignore
        result = self.calc_rpn()

        if type(result) is int:
            return result
        return result

    def calc_rpn(self):
        output = []
        operators = self.ctx.operators
        ops = operators.keys()
        func_map = self.ctx.functions
        funcs = func_map.keys()
        rpn = self.rpns
        var_map = self.ctx.variables

        def call_operator(operator: str):
            try:
                a, b = output[-2], output[-1]
            except IndexError:
                raise SyntaxError("Unfinished line")
            del output[-2:]

            op_to_run: types.Operator = operators[operator] #type: ignore
            to_app = self.ctx.cache.get((operator, (a, b))) #type: ignore
            if not to_app:
                for valid in op_to_run.validators:
                    valid(a, b, op=operator)

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
                            f"Operation {a} {operator} {b} will lead to at leat {n_digits} digits out of maximum of {cst.MAXIMUM_DIGITS}")

                    if n_digits > cst.MAXIMUM_DIGITS_WARNING:
                        self.logger.warning(
                            f"Operation {a} {operator} {b} will lead to at least {n_digits} digits(warning set on {cst.MAXIMUM_DIGITS_WARNING})")
                try:
                    to_app = decimal.Decimal(op_to_run.callable_function(a, b))
                    if utils.check_is_integer(to_app):
                        to_app = int(to_app)  # type: ignore
                except decimal.InvalidOperation:
                    raise TypeError(f"Cannot apply {operator} to '{a}' and '{b}'")
                except TypeError:
                    to_app = op_to_run.callable_function(a, b, ctx = self.ctx)
                    to_app = decimal.Decimal(to_app)
                    if utils.check_is_integer(to_app):
                        to_app = int(to_app)  # type: ignore
                self.ctx.cache[(operator, (a, b))] = to_app #type: ignore
            output.append(to_app)

        for i in range(len(rpn)):
            t = rpn[i]
            if type(t) is list:
                continue
            if t in var_map.keys():
                var = var_map[t]
                result = None
                if not var.local:
                    result = self.ctx.cache.get((t, ()))
                    self.ctx.outer_names_buffer.append(t)
                if not result:
                    cls = Calculator(ctx = self.ctx)
                    result = cls.calc(var.value)
                    self.ctx.cache[(t, ())] = result
                if not var.local:
                    self.ctx.outer_names_buffer.pop()
                to_app = decimal.Decimal(result)
                if utils.check_is_integer(to_app):
                    to_app = int(to_app)  # type: ignore
                output.append(to_app)
            elif t in ops:
                call_operator(t)
            elif t in funcs:

                current_tokens = []  # type: ignore
                current_args = []
                for arg in rpn[i + 1]:  # type: ignore
                    if arg == ',':
                        new_calc = Calculator(current_tokens, ctx=self.ctx)
                        arg_val = new_calc.calc_rpn()
                        current_args.append(arg_val)
                        current_tokens.clear()
                    else:
                        current_tokens.append(arg)

                new_calc = Calculator(current_tokens, ctx = self.ctx, )
                arg_val = new_calc.calc_rpn()
                current_args.append(arg_val)
                current_tokens.clear()
                func = func_map[t]  # type: ignore
                res = self.ctx.cache.get((t, tuple(current_args)), None)
                for val in func.validators:
                    try:
                        val(*current_args, op=t)
                    except Exception:
                        raise
                if not res:
                    self.logger.debug(f"Calling function {t}({current_args})")
                    try:
                        res = func.callable_function(*current_args, ctx=self.ctx)
                    except TypeError:
                        res = func.callable_function(*current_args)
                    self.ctx.cache[(t, tuple(current_args))] = res
                output.append(res)

            else:
                output.append(t)
        return output[0]
