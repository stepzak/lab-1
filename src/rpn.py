import decimal

from src.extra import utils
from src.extra.types import Context
from src.extra.exceptions import InvalidTokenError

@utils.init_default_ctx
class ConverterRPN:
    def __init__(self, ctx: Context):
        self.ctx = ctx
    def rpn(self, tokens: list) -> list:
        """
        Converts list of tokens to RPN and then calcs their value
        :return: value of RPN-converted tokens
        """
        if not tokens:
            return [" "]
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
                    last_func[2].extend(self.rpn(last_func[1]) + [","])
                    last_func[1].clear()

                elif t == "]":

                    last_func[2].extend(self.rpn(last_func[1]))

                    stack_functions.pop()
                    if len(stack_functions):
                        stack_functions[-1][2].extend([last_func[0], last_func[2]])
                    else:
                        output.extend([last_func[0], last_func[2]])

                else:
                    last_func[1].append(t)
            else:
                try:
                    to_app = decimal.Decimal(t)
                    if utils.check_is_integer(to_app):
                        to_app = int(to_app)  # type: ignore
                    output.append(to_app)  # type: ignore
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
                            while (-2 * (
                                    prev_op.is_right is True and cur_op.is_right is True) + 1) * prev_op.priority >= cur_op.priority:
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
