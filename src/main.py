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
    :return: result of an expression
    """

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
    stack: list[str] = []  # list of operations
    for t in tokens:
        try:
            output.append(float(t))
            continue
        except ValueError:
            if t in ops:
                priority = operators[t][0]
                if len(stack):
                    if stack[-1]=="(":
                        stack.append(t)
                        continue
                    while operators[stack[-1]][0]>=priority:
                        op = stack.pop()
                        a, b = output[-2], output[-1]
                        if op == "//" or op == "%":
                            if not (check_is_integer(a) and check_is_integer(b)):
                                return f"Cannot apply '{op}' to {a} and {b}"
                        del output[-2:]
                        output.append(operators[op][1](a, b))
                        if not len(stack) or stack[-1] == "(":
                            break
                stack.append(t)
            elif t == "(":
                stack.append(t)
            elif t == ")":
                while stack[-1]!="(":
                    op = stack.pop()
                    a, b = output[-2], output[-1]
                    del output[-2:]
                    output.append(operators[op][1](a, b))
                    if not len(stack) or stack[-1] == "(":
                        break
                stack.pop()
    for op in stack[::-1]:
        a, b = output[-2], output[-1]
        if op == "//" or op == "%":
            if not (check_is_integer(a) and check_is_integer(b)):
                return f"Cannot apply '{op}' to {a} and {b}"
        elif op == "/" and b == 0:
            return "Cannot divide by zero"
        del output[-2:]

        output.append(operators[op][1](a, b))

    return output[0]


def tokenize(expression: str) -> list[str] | str:
    """
    Tokenizes the expression\n
    :param expression -- raw mathematical expression needed to be tokenized(no spaces)\n
    :returns list of tokens
    """
    tokens: list[str] = []
    is_digit: bool = False #is the last symbol digit(flag)

    for s in expression:
        if s.isdigit() or s==".": #if current symbol is digit
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

        elif not is_digit and s not in "()" and tokens[-1] not in "()":

            return "Invalid syntax: 2 non-unary operations one after other"

        else:
            tokens.append(s) #appending the other tokens
        is_digit = False

    return tokens

def calc(expression: str) -> float | str:
    tokens = tokenize(expression)
    if isinstance(tokens, str):
        return tokens
    result = rpn_and_calc(tokens)
    return result



def main():
    """
    Entry point for application. Checks if expression is OK and removes all the spaces, then calls tokenize() and calc() functions
    """
    while True:
        expression: str = input("Enter the expression to calculate: ")
        expression = expression.replace(" ", "")  # remove spaces
        if not (expression[0].isdigit() or expression[0] in '-+'): #check if beginning is OK
            logger.error("Must begin with unary operation or number")
            continue
        #parentheses check
        parentheses_total: int = 0

        for s in expression:
            if s=="(":
                parentheses_total+=1
            elif s==")":
                parentheses_total-=1
            if parentheses_total<0:
                break
        if parentheses_total!=0:
            logger.error("Invalid parenthesis syntax")
            continue

        result = calc(expression)
        if not isinstance(result, float):
            logger.error(result)
            continue

        logger.info(f"{expression}={result}")


if __name__ == "__main__":
    main()
