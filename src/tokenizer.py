import string
import logging

import src.extra.exceptions as ex_exc
from extra import utils
from extra.utils import log_exception
from src.extra.types import Context
from src.extra.utils import get_next_token, get_previous_token
from src.validator import CompiledValidExpression

@utils.init_default_ctx
class Tokenizer:
    def __init__(self, ctx: Context, logger: logging.Logger):
        self.ctx = ctx
        self.logger = logger or logging.getLogger(__name__)

    @log_exception
    def tokenize(self, compiled_expression: CompiledValidExpression) -> list[str]:
        """
        Tokenizes the expression
        :param compiled_expression: raw mathematical expression needed to be tokenized(after compilation and scope defining)
        :return: list of tokens
        """

        expression = compiled_expression.expression.strip()
        func_map = self.ctx.functions
        var_map = self.ctx.variables
        op_map = self.ctx.operators
        outer_names_buffer = self.ctx.outer_names_buffer
        tokens: list[str] = ["  "]

        def place_token(token: str):
            """
            For extra conditions in the future
            """
            tokens.append(token)


        digitable = list(func_map.keys()) + list(string.digits) + list(var_map.keys()) + [".", "e"]

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

        known_tokens = digitable + list(op_map.keys()) + ["[", "]", "(", ")", ',', '.', 'e', ' ', '', "  "]
        current_functions: list[tuple[list[str], list[int], list[
            int]]] = []  # list of parsing functions. ({func_symbol}, {arg_count}, {parenthesis_count})
        for expr_ind in range(len(expression)):
            s = expression[expr_ind]
            if len(tokens)>=2:
                rec_error = False
                if tokens[-2] not in op_map.keys():
                    rec_error = tokens[-2] in outer_names_buffer
                elif len(outer_names_buffer):
                    rec_error = tokens[-2] == outer_names_buffer[-1]

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
                        if check_is_digit(tokens[-3]) and check_is_digit(tokens[-1]):
                            raise SyntaxError(f"Missed operation between {tokens[-3]} and {tokens[-1]}")
            if s == " " and not tokens[-1].isspace():
                tokens.append(" ")
                continue

            def filter_func(x: str):
                return x.startswith(tokens[-1]) or x.startswith(s)

            multi_symbols = list(filter(filter_func, op_map.keys()))
            multi_symbols += list(filter(filter_func, func_map.keys()))
            if var_map:
                multi_symbols += list(filter(filter_func, var_map.keys()))

            extra_check_for_unary = True and len(tokens)>1
            if s in "+-" and len(tokens)>1:
                try:
                    extra_check_for_unary = not check_is_digit(get_next_token(expression, expr_ind))
                except IndexError:
                    raise ex_exc.InvalidTokenError(f"Unfinished line: operation '{s}' has no second operand",
                                                   exc_type="invalid_token")

            if any(tokens[-1] + s in ms for ms in multi_symbols) and len(multi_symbols):
                tokens[-1] += s

            elif any(s in ms and s for ms in multi_symbols) and len(multi_symbols) and extra_check_for_unary:
                    place_token(s)
            else:
                if tokens[-1] in func_map.keys():
                    current_functions.append(([tokens[-1]], [0], [0]))

                elif tokens[-1] not in known_tokens and not check_is_digit(tokens[-1]):
                    raise ex_exc.InvalidTokenError(f"Unknown token: '{tokens[-1]}'", exc_type="unknown_token")
                if s.isdigit() or s in (".", "e"):  # if current symbol is digit
                    if not check_is_digit(tokens[-1]):  # if previous one was not a digit we append the symbol as the first digit of a number
                        place_token(s)
                        if check_is_digit(tokens[-1]) and check_is_digit(tokens[-2]):
                            for digit in expression[expr_ind + 1:]:
                                if check_is_digit(tokens[-1] + digit) or digit in [".", "e"]:
                                    tokens[-1] += digit
                                else:
                                    break

                            raise SyntaxError(f"Two numbers without an operator: '{tokens[-2]}' '{tokens[-1]}'")
                    else:
                        tokens[-1] += s  # if the previous one was a digit, we concatenate current token with symbol

                    continue  # continues the loop(to eliminate extra checks)
                elif s == "-":
                    res = get_previous_token(tokens, expr_ind)
                    if not check_is_digit(res):  # context management: if previous one was not a digit, it is a unary "-"
                        tokens.extend(("-1", "*"))
                        if res not in "()[] " and not res.isspace():
                            self.logger.warning(f"Two operations({res}-), '-' is unary: one after other detected")
                    else:
                        place_token("-")  # else it is a substraction
                elif s == "+":
                    res = get_previous_token(tokens, expr_ind)
                    if not check_is_digit(res) and res not in ")]" and not res.isspace():
                        self.logger.warning(f"Two operations({res}+), '+' is unary: one after other detected")
                    elif check_is_digit(res) or res in ")]":
                        tokens.append("+")
                    continue

                elif not check_is_digit(tokens[-1]) and s in op_map.keys():

                    for i in range(1, len(tokens) - 1):
                        if check_is_digit(tokens[-i]):
                            break
                        elif tokens[-i] in op_map.keys():
                            raise SyntaxError(f"Two operations one after other: {tokens[-i]}{s}")
                    place_token(s)
                elif s == ")":
                    for i in range(1, len(tokens)):
                        if check_is_digit(tokens[-i]):
                            break
                        elif tokens[-i] in op_map.keys():
                            raise SyntaxError(f"Unfinished line: operation '{tokens[-i]}' has no second operand")
                        elif tokens[-i] == "(":
                            if not len(current_functions):
                                raise ex_exc.InvalidParenthesisError("Empty parenthesis", exc_type="empty")
                            if current_functions[-1][2][0] != 0:
                                raise ex_exc.InvalidParenthesisError("Empty parenthesis", exc_type="empty")
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
                            min_func_args = func_map[cur_func[0][0][:-1]].min_args
                            if check_is_digit(tokens[-2]):
                                cur_func[1][0] += 1
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
                        max_func_args = func_map[cur_func[0][0][:-1]].max_args
                        if max_func_args != -1 and max_func_args < args:
                            func_name = cur_func[0][0][:-1]

                            raise TypeError(
                                f"TypeError: {func_name} requires maximum of {max_func_args} arguments but at least {cur_func[1]} were given")

        self.logger.debug(f"{tokens=}")

        if outer_names_buffer:
            for ind in range(1, len(tokens)):
                t = tokens[-ind]
                if t in outer_names_buffer:
                    raise RecursionError(f"Name '{t}' defined with itself({expression}). Recursion is not (yet) supported")

        if len(tokens)>=3:

            if tokens[-2] == " ":
                if check_is_digit(tokens[-1]) and check_is_digit(tokens[-3]):
                    raise SyntaxError(f"Two numbers without an operator: '{tokens[-3]}' '{tokens[-1]}'")

            for t in tokens[::-1]:
                if t in op_map.keys() and t not in "+-":
                    raise ex_exc.InvalidTokenError(f"Unfinished line: operation '{t}' has no second operand",
                                                   exc_type="invalid_token")
                elif check_is_digit(t):
                    break

            for t in tokens:
                if t in op_map.keys() and t not in "+-":
                    raise ex_exc.InvalidTokenError(f"Unfinished line: operation '{t}' has no first operand",
                                                   exc_type="invalid_token")
                elif check_is_digit(t):
                    break

        return tokens
