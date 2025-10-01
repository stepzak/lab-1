import decimal
import logging
import string
from decimal import getcontext
from sys import stdout
import constants as cst # type: ignore
from extra.exceptions import InvalidTokenError #type: ignore
from extra.utils import log_exceptions, round_decimal, check_is_integer #type: ignore
from validator import PreCompiledValidExpression, CompiledValidExpression  #type: ignore
from compiler import CompiledExpression #type: ignore
from vars import FUNCTIONS, FUNCTIONS_CALLABLE_ENUM, OPERATORS, FUNCTIONS_ARGS #type: ignore

getcontext().prec = cst.PRECISION

logger = logging.getLogger(__name__)


class CustomFunctionExecutor:
    def __init__(self, indexed_vars: list[tuple[str, str | None]],  expression: str):
        self.expression = expression
        self.indexed_vars = indexed_vars

    @log_exceptions(logger = logger)
    def execute_function(self, *args, var_map: dict) -> decimal.Decimal:
        var_map = var_map.copy()
        for i in range(len(self.indexed_vars)):
            try:
                var_map.update({self.indexed_vars[i][0]: str(args[i])})
            except IndexError:
                var_map.update({self.indexed_vars[i][0]: self.indexed_vars[i][1]})
        return calc(self.expression, var_map)

@log_exceptions(logger = logger)
def rpn_and_calc(tokens: list[str], var_map: dict[str, str]) -> decimal.Decimal | None:
    """
    Converts list of tokens to RPN and then calcs to value
    :param tokens: list of tokens
    :param var_map: dict of variables
    :return: value of RPN-converted tokens
    """
    operators = OPERATORS
    ops = operators.keys()

    output: list[decimal.Decimal] = []
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
            output.append(op_to_run[1](a, b))
        except decimal.InvalidOperation:
            raise TypeError(f"Cannot apply {operator} to '{a}' and '{b}'")

    for t in tokens:
        if t in ['', " "]:
            continue

        if t in FUNCTIONS:
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
                func = FUNCTIONS_CALLABLE_ENUM[last_func[0]][0]
                validators = FUNCTIONS_CALLABLE_ENUM[last_func[0]][1]
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
                output.append(decimal.Decimal(t))
                continue
            except decimal.InvalidOperation:

                if t in var_map.keys():
                    output.append(decimal.Decimal(calc(var_map[t], var_map)))
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
                        while (-1*(prev_op[2] is True and cur_op[2] is True))*prev_op[0]>=cur_op[0]:
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
                    raise SyntaxError(f"Unknown token: '{t}'")

    for op in stack_ops[::-1]:
        call_operator(op)

    if len(output):
        return output[0]
    return None


@log_exceptions(logger=logger)
def tokenize(compiled_expression: CompiledValidExpression, var_map: dict[str, str]) -> list[str]:
    """
    Tokenizes the expression
    :param compiled_expression -- raw mathematical expression needed to be tokenized(after compilation)
    :param var_map: map of variables: {"var_name": "var_expression"}
    :return: list of tokens
    """

    expression = compiled_expression.expression

    tokens: list[str] = [""]
    is_digit: bool = False  #is the last symbol digit(flag)
    digitable = FUNCTIONS+list(var_map.keys())+['']+list(string.digits)
    known_tokens = digitable+list(OPERATORS.keys())+["[", "]", "(", ")", ',', '.', 'e']
    current_functions: list[tuple[list[str], list[int], list[int]]] = []  #list of parsing functions. ({func_symbol}, {arg_count}, {parenthesis_count})
    for s in expression:
        #shift = len(tokens[-1])
        multi_symbols = list(filter(lambda x: x.startswith(tokens[-1]), OPERATORS.keys()))
        multi_symbols+=list(filter(lambda x: x.startswith(tokens[-1]), FUNCTIONS_CALLABLE_ENUM.keys()))
        multi_symbols+=list(filter(lambda x: x.startswith(tokens[-1]), var_map.keys()))
        for multi_symbol in multi_symbols:
            if multi_symbol.startswith(tokens[-1]+s):
                tokens[-1]+=s
                break
        else:
            if tokens[-1] in FUNCTIONS:
                current_functions.append(([tokens[-1]], [0], [0]))

            elif tokens[-1] not in known_tokens and not is_digit:
                raise InvalidTokenError(f"Unknown token: '{tokens[-1]}'", exc_type="unknown_token")
            if tokens[-1] in digitable:
                for func in current_functions:
                    if func[1][0] == 0:
                        func[1][0] += 1
            if s.isdigit() or s in (".", "e"):  #if current symbol is digit
                if not is_digit:  #if previous one was not a digit we append the symbol as the first digit of a number
                    tokens.append(s)
                    is_digit = True  #flag
                else:
                    tokens[-1] += s  #if the previous one was a digit, we concatenate current token with symbol

                continue  #continues the loop(to eliminate extra checks)
            elif s == "-":
                if not is_digit:  #context management: if previous one was not a digit, it is a unary "-"
                    tokens.extend(("-1", "*"))
                    logger.warning("Two unary operations(-) one after other detected")
                else:
                    tokens.append("-")  #else it is a substraction
            elif s == "+" and not is_digit:
                tokens.append("+")
                logger.warning("Two unary operations(+) one after other detected")
                continue

            elif tokens[-1] not in digitable and s in OPERATORS.keys() and tokens[-1] not in "[]()":
                raise TypeError("2 operations one after other")


            else:
                tokens.append(s)  #appending the other tokens
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
                        min_func_args = FUNCTIONS_ARGS[cur_func[0][0][:-1]][0]
                        if min_func_args != -1 and min_func_args > cur_func[1][0]:
                            func_name = cur_func[0][0][:-1]
                            raise TypeError(f"TypeError: {func_name} requires minimum  of {min_func_args} arguments but {cur_func[1][0]} were given")

                        tokens[-1] = "]"

                        current_functions.pop()
                        continue

                elif s == ",":
                    cur_func[1][0] += 1
                    args: int = cur_func[1][0]
                    max_func_args = FUNCTIONS_ARGS[cur_func[0][0][:-1]][1]
                    if max_func_args != -1 and max_func_args < args:
                        func_name = cur_func[0][0][:-1]

                        raise TypeError(f"TypeError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")
            if s not in "()":
                is_digit = False
    logger.debug(f"{tokens=}")
    if tokens[-1] in "+-**//%":
        raise SyntaxError("Unfinished line")
    return tokens


@log_exceptions(logger = logger)
def calc(expr: str, var_map = None) -> decimal.Decimal | int | None:
    """
    Calculates value of the expression
    :param expr: Expression to calculate
    :param var_map: map of variables {"var_name": "var_expression"}
    :return: (error code, value of the expression). Value can be returned as None if an error occurred during calculation process
    """
    global FUNCTIONS_CALLABLE_ENUM, FUNCTIONS, FUNCTIONS_ARGS
    pre_compiled = PreCompiledValidExpression(expr)
    expression = CompiledExpression(pre_compiled.expression, var_map)
    if not var_map:
        var_map = expression.var_map
        RESET_CALLABLE_ENUM = FUNCTIONS_CALLABLE_ENUM.copy()
        RESET_FUNCTIONS_ENUM = FUNCTIONS.copy()
        RESET_FUNCTIONS_ARGS = FUNCTIONS_ARGS.copy()

    func_map = expression.func_map
    for k, v in func_map.items():
        obj = CustomFunctionExecutor(v[0], v[1])
        FUNCTIONS_CALLABLE_ENUM[k] = (obj.execute_function, None)
        FUNCTIONS_ARGS[k] = (v[2], v[3])
    FUNCTIONS = list(FUNCTIONS_CALLABLE_ENUM.keys())
    compiled_expression = CompiledValidExpression(expression)


    logger.debug(f"{compiled_expression.expression=}")
    tokens = tokenize(compiled_expression, var_map)
    if not tokens:
        return None
    result = rpn_and_calc(tokens, var_map)
    if not var_map:
        FUNCTIONS = RESET_FUNCTIONS_ENUM.copy()
        FUNCTIONS_CALLABLE_ENUM = RESET_CALLABLE_ENUM.copy()
        FUNCTIONS_ARGS = RESET_FUNCTIONS_ARGS.copy()

    if not result:
        return result
    if type(result) is int:
        return result
    result = round_decimal(result)
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
            result = calc(expression)
        except Exception:
            result = None

        logger.info(f"{expression} = {result}")


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
