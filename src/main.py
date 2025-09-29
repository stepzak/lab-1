import decimal
import logging
from decimal import getcontext
from functools import wraps
from sys import stdout
from typing import Callable

import constants as cst
from src.checks import check_parenthesis, check_vars, check_for_forbidden_symbols, check_is_integer
from src.compiles import compile_functions, compile_vars


getcontext().prec = cst.PRECISION

def round_decimal(dec: decimal.Decimal, n_digits: int = cst.ROUNDING_DIGITS, rounding = cst.ROUNDING):
    quantizer = decimal.Decimal('1.' + '0' * n_digits)
    return dec.quantize(quantizer, rounding=rounding)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.DEBUG,
    handlers=[
        logging.FileHandler(cst.LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(stdout),
    ],
    format = cst.FORMAT
)

def log_exceptions(func):
    """Decorator to automatically log exceptions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
    return wrapper


@log_exceptions
def rpn_and_calc(tokens: list[str]) -> decimal.Decimal:
    """
    Converts list of tokens to RPN and then calcs to value
    :param tokens: list of tokens
    :return: value of RPN-converted tokens
    """
    if '' in tokens:
        tokens = tokens[0:]
    operators: dict[str, tuple[int, Callable[[decimal.Decimal, decimal.Decimal], decimal.Decimal]] ]= {
        "+": (0, lambda x, y: x+y),
        "-": (0, lambda x, y: x-y),
        "*": (1, lambda x, y: x*y),
        "/": (1, lambda x, y: x/y),
        "//": (1, lambda x, y: x//y),
        "%": (1, lambda x, y: x%y),
        "**": (2, lambda x, y: x**y)
    }

    ops = operators.keys()

    output: list[decimal.Decimal] = []
    stack_ops: list[str] = []  # list of operations
    stack_functions: list[tuple[str, list, list]] = [] #{function}, {tokens}, {args}
    for t in tokens:
        if t == '':
            continue

        if t in cst.FUNCTIONS:
            stack_functions.append((t, [], []))
            continue

        elif len(stack_functions):
            last_func = stack_functions[-1]

            if t == "[":
                continue

            elif t == ',':
                last_func[2].append(rpn_and_calc(last_func[1]))
                last_func[1].clear()


            elif t == "]":
                last_func[2].append(rpn_and_calc(last_func[1]))
                if last_func[0]==cst.FUNCTIONS_SYMBOLS_ENUM["pow"]:
                    for i in range(len(last_func[2])):
                        if check_is_integer(last_func[2][i]):
                            last_func[2][i] = int(last_func[2][i])
                        elif len(last_func[2])==3:
                            raise TypeError("pow() 3rd argument not allowed unless all arguments are integers")
                if cst.SYMBOLS_FUNCTIONS_ENUM[last_func[0]]!="sqrt":
                    res = cst.SYMBOLS_CALLABLE_ENUM[last_func[0]](*last_func[2])
                else:
                    res = last_func[2][0].sqrt()
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
                if t in ops:
                    priority = operators[t][0]
                    if len(stack_ops):
                        if stack_ops[-1]== "(":
                            stack_ops.append(t)
                            continue
                        while operators[stack_ops[-1]][0]>=priority:
                            op = stack_ops.pop()
                            a, b = output[-2], output[-1]
                            if op == "//" or op == "%":
                                if not (check_is_integer(a) and check_is_integer(b)):
                                    raise TypeError(f"Cannot apply '{op}' to {a} and {b}")
                            del output[-2:]
                            output.append(operators[op][1](a, b))
                            if not len(stack_ops) or stack_ops[-1] == "(":
                                break
                    stack_ops.append(t)
                elif t == "(":
                    stack_ops.append(t)
                elif t == ")":
                    while stack_ops[-1]!= "(":
                        op = stack_ops.pop()
                        a, b = output[-2], output[-1]
                        del output[-2:]
                        output.append(operators[op][1](a, b))
                        if not len(stack_ops) or stack_ops[-1] == "(":
                            break
                    stack_ops.pop()
                else:
                    raise SyntaxError(f"Unknown token: '{t}'")

    for op in stack_ops[::-1]:
        a, b = output[-2], output[-1]
        if op == "//" or op == "%":
            if not (check_is_integer(a) and check_is_integer(b)):
                raise TypeError(f"Cannot apply '{op}' to {a} and {b}")
        elif op == "/" and b == 0:
            raise ZeroDivisionError
        del output[-2:]
        try:
            output.append(operators[op][1](a, b))
        except decimal.InvalidOperation:
            raise TypeError(f"Cannot apply '{op}' to {a} and {b}")

    return output[0]

@log_exceptions
def tokenize(expression: str) -> list[str]:
    """
    Tokenizes the expression
    :param expression -- raw mathematical expression needed to be tokenized(no spaces)
    :return: list of tokens
    """
    tokens: list[str] = [""]
    is_digit: bool = False #is the last symbol digit(flag)


    current_functions: list[tuple[ list[str], list[int], list[int] ]] = [] #list of parsing functions. ({func_symbol}, {arg_count}, {parenthesis_count})
    for s in expression:

        if s not in cst.AVAILABLE_SYMBOLS:
            raise SyntaxError(f"Unknown token found: '{s}'")

        if s.isdigit() or s==".": #if current symbol is digit
            for func in current_functions:
                if func[1][0]==0:
                    func[1][0]+=1
            if not is_digit: #if previous one was not a digit we append the symbol as the first digit of a number
                tokens.append(s)
                is_digit=True #flag
            else:
                tokens[-1]+=s #if the previous one was a digit, we concatenate current token with symbol

            continue #continues the loop(to eliminate extra checks)

        elif s == "-":
            if not is_digit: #context management: if previous one was not a digit, it is a unary "-"
                tokens.extend(("-1", "*"))
            else:
                tokens.append("-") #else it is a substraction
        elif s == "+" and not is_digit:
            continue

        elif (s=="*" or s == "/") and tokens[-1] == s:
            tokens[-1]+=s
            continue

        elif s in cst.FUNCTIONS:
            current_functions.append(([s], [0], [0] ))
            tokens.append(s)
            continue


        elif not is_digit and s not in "()" and tokens[-1] not in "(),[]" and s not in cst.FUNCTIONS:
            raise TypeError("2 binary operations one after other")

        elif tokens[-1] in cst.FUNCTIONS and s!="(":
            raise SyntaxError("'(' must follow after a function")

        else:
            tokens.append(s) #appending the other tokens
        if len(current_functions)>0:
            cur_func = current_functions[-1]
            if s == "(":
                if "[" not in cur_func[0][0]:
                    cur_func[0][0]+="["

                    tokens[-1] = "["
                else:
                    current_functions[-1][2][0]+=1
                continue
            elif s==")":
                cur_func[2][0]-=1
                if cur_func[2][0]<0:
                    min_func_args = cst.FUNCTIONS_ARGS[cur_func[0][0][:-1]][0]
                    if min_func_args != -1 and min_func_args > cur_func[1][0]:
                        func_name = cst.SYMBOLS_FUNCTIONS_ENUM[cur_func[0][0][:-1]]
                        raise TypeError(f"TypeError: {func_name} requires minimum  of {min_func_args} arguments but {cur_func[1]} were given")

                    tokens[-1]="]"

                    current_functions.pop()
                    continue

            elif s == ",":
                cur_func[1][0]+=1
                args: int = cur_func[1][0]
                max_func_args = cst.FUNCTIONS_ARGS[cur_func[0][0][:-1]][1]
                if max_func_args!=-1 and max_func_args<args:
                    func_name = cst.SYMBOLS_FUNCTIONS_ENUM[cur_func[0][0][:-1]]

                    raise TypeError(f"TypeError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")
        if s not in "()":
            is_digit = False
    logger.debug(f"{tokens=}")
    return tokens

@log_exceptions
def calc(expression: str) -> decimal.Decimal | int:
    """
    Calculates value of the expression
    :param expression: Expression to calculate
    :return: (error code, value of the expression). Value can be returned as None if an error occurred during calculation process
    """

    forbidden = check_for_forbidden_symbols(expression)
    if len(forbidden):
        raise SyntaxError(f"Forbidden symbol detected: {forbidden}")

    expression = expression.replace(" =", "=")
    expression = compile_functions(expression)

    if ";" in expression:
        checked_vars = check_vars(expression)
        if len(checked_vars):
            raise SyntaxError(f"Variable '{checked_vars}' overshadows default function name")
        expression = compile_vars(expression)

    expression = expression.replace(" ", "")

    if not (expression[0].isdigit() or expression[0] in '-+('+''.join(cst.FUNCTIONS)):  # check if beginning is OK
        raise SyntaxError("Must begin with unary operation or number")

    if not check_parenthesis(expression):
        raise SyntaxError("Invalid parenthesis syntax")

    tokens = tokenize(expression)

    result = rpn_and_calc(tokens)
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


        expression: str = input("Enter the expression to calculate: ")

        result = calc(expression)

        logger.info(f"{expression}={result}")


if __name__ == "__main__":
    main()
