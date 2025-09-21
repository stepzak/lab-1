import logging
from sys import stdout
from typing import Callable

import constants as cst

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    handlers=[
        logging.FileHandler(cst.LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(stdout),
    ],
    format = cst.FORMAT
)

def check_is_integer(x: str | float) -> bool:
    """

    :param x: string or float to check if it is an integer
    :return: True if it is; False if it is not
    """
    return float(x)==int(x)

def rpn_and_calc(tokens: list[str]) -> float | str:
    """
    Converts list of tokens to RPN and then calcs to value
    :param tokens: list of tokens
    :return: result of the expression
    """
    if '' in tokens:
        tokens = tokens[0:]
    operators: dict[str, tuple[int, Callable[[float, float], float]] ]= {
        "+": (0, lambda a, b: a+b),
        "-": (0, lambda a, b: a-b),
        "*": (1, lambda a, b: a*b),
        "/": (1, lambda a, b: a/b),
        "//": (1, lambda a, b: a//b),
        "%": (1, lambda a, b: a%b),
        "**": (2, lambda a, b: a**b)
    }

    ops = operators.keys()

    output: list[float] = []
    stack_ops: list[str] = []  # list of operations
    stack_functions: list[tuple[str, list, list]] = [] #{function}, {tokens}, {args}
    for t in tokens:

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
                continue

            elif t == "]":
                last_func[2].append(rpn_and_calc(last_func[1]))
                if last_func[0]==cst.FUNCTIONS_SYMBOLS_ENUM["pow"]:
                    for i in range(len(last_func[2])):
                        if last_func[2][i].is_integer():
                            last_func[2][i] = int(last_func[2][i])
                        elif len(last_func[2])==3:
                            logger.error("pow() 3rd argument not allowed unless all arguments are integers")
                            return "undefined"

                res = cst.SYMBOLS_CALLABLE_ENUM[last_func[0]](*last_func[2])
                stack_functions.pop()
                if len(stack_functions):
                    stack_functions[-1][1].append(str(res))
                else:
                    output.append(res)
                continue

            else:
                last_func[1].append(t)
                continue

        try:
            output.append(float(t))
            continue
        except ValueError:

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
                                logger.error( f"Cannot apply '{op}' to {a} and {b}")
                                return "undefined"
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
    for op in stack_ops[::-1]:
        a, b = output[-2], output[-1]
        if op == "//" or op == "%":
            if not (check_is_integer(a) and check_is_integer(b)):
                logger.error( f"Cannot apply '{op}' to {a} and {b}")
                return "undefined"
        elif op == "/" and b == 0:
            logger.error("Cannot divide by zero")
            return "undefined"
        del output[-2:]

        output.append(operators[op][1](a, b))

    return output[0]


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
            if not is_digit: #context management: if previous one was not a digit, it is an unary "-"
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
            if s!=cst.FUNCTIONS_SYMBOLS_ENUM["sqrt"]:
                tokens.append(s)
            continue


        elif not is_digit and s not in "()" and tokens[-1] not in "(),[]" and s not in cst.FUNCTIONS:
            logger.error("Invalid syntax: 2 non-unary operations one after other")
            return []

        elif tokens[-1] in cst.FUNCTIONS and s!="(":
            logger.error("Invalid syntax: '(' must follow after a function")
            return []

        else:
            tokens.append(s) #appending the other tokens
        if len(current_functions)>0:
            cur_func = current_functions[-1]
            if s == "(":
                if "[" not in cur_func[0][0]:
                    cur_func[0][0]+="["
                    if cur_func[0][0]!=cst.FUNCTIONS_SYMBOLS_ENUM["sqrt"]+ "[":
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
                        logger.error(
                            f"ArgumentError: {func_name} requires minimum  of {min_func_args} arguments but {cur_func[1]} were given")
                        return []
                    if cur_func[0][0]!=cst.FUNCTIONS_SYMBOLS_ENUM["sqrt"]+"[":
                        tokens[-1]="]"

                    else:
                        tokens.extend(["**", "0.5"])
                    current_functions.pop()
                    continue

            elif s == ",":
                cur_func[1][0]+=1
                args: int = cur_func[1][0]
                max_func_args = cst.FUNCTIONS_ARGS[cur_func[0][0][:-1]][1]
                if max_func_args!=-1 and max_func_args<args:
                    func_name = cst.SYMBOLS_FUNCTIONS_ENUM[cur_func[0][0][:-1]]
                    logger.error(f"ArgumentError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")
                    return []
        if s not in "()":
            is_digit = False

    return tokens

def calc(expression: str) -> float | str:
    expression = expression.replace(" =", "=")
    expression = compile_functions(expression)

    if ";" in expression:
        if not check_vars(expression):
            return "undefined"
        expression = compile_vars(expression)
    expression = expression.replace(" ", "")  # remove spaces
    if not (expression[0].isdigit() or expression[0] in '-+('+''.join(cst.FUNCTIONS)):  # check if beginning is OK
        logger.exception(SyntaxError("Must begin with unary operation or number"))
        return "undefined"
    # parentheses check
    if not check_parenthesis(expression):
        return "undefined"

    tokens = tokenize(expression)
    if len(tokens)==0:
        return "undefined"
    try:
        result = rpn_and_calc(tokens)
    except (TypeError, ZeroDivisionError) as ex:
        logger.error(ex)
        result = "undefined"
    return result


def check_parenthesis(expression: str) -> bool:
    """
    Checks if parenthesis syntax is OK
    :param expression: expression to check
    :return: True if OK else False
    """
    parentheses_total: int = 0

    for s in expression:
        if s == "(":
            parentheses_total += 1
        elif s == ")":
            parentheses_total -= 1
        if parentheses_total < 0:
            logger.error(SyntaxError("Invalid parenthesis syntax"))
            return False
    res = parentheses_total == 0
    if not res:
        logger.error(SyntaxError("Invalid parenthesis syntax"))
    return res

def check_vars(expression: str) -> bool:
    """
    Checks if vars do not overshadow default function names(DFN)
    :param expression: expression to check. Syntax: let x = ...; let y = ...; x+y
    :return: True if vars do not overshadow DFN else False
    """
    splitted = expression.split(";")

    for var in splitted[:-1]:
        checks = ("let max=", "let min=", "let abs=", "let sqrt=", "let pow=")
        if any(c in var for c in checks):
            logger.error(f"Variable '{var[4:].split('=')[0]}' overshadows default function name")
            return False
    return True

def compile_vars(expression: str) -> str:
    """
    Replaces variables with their values
    :param expression: Expression needed to be compiled
    :return: Var-compiled expression
    """
    splitted = expression.split(";")
    var_map: list[list[str]] = []
    for var in splitted[:-1]:
        var_name, var_val = var[4:].split("=")
        var_name = var_name.replace(" ", "")
        var_map.append([var_name, var_val])
    var_map = list( sorted(var_map, key=lambda item: len(item[0])))
    for i in range(len(var_map)):
        k, v = var_map[i]
        expression = expression.replace(k, v)
        for x in var_map[i+1:]:
            x[1] = x[1].replace(k, v)
    return expression.split(";")[-1]


def compile_functions(expression: str) -> str:
    """
    Compiles functions to single symbols(for an easier parser) by rules
    set in FUNCTIONS_SYMBOLS_ENUM(constants.py)
    :param expression: Expression needed to be compiled
    :return: Compiled expression
    """

    for func, symb in cst.FUNCTIONS_SYMBOLS_ENUM.items():
        expression = expression.replace(func+"(", symb+"(")
    return expression


def main():
    """
    Entry point for application. Checks if expression is OK and removes all the spaces, then calls tokenize() and calc() functions
    """
    while True:


        expression: str = input("Enter the expression to calculate: ")

        result = calc(expression)

        logger.info(f"{expression}={result}")


if __name__ == "__main__":
    main()
