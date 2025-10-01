from extra.exceptions import InvalidTokenError, VariableOvershadowError
from extra.utils import CallAllMethods
def parse_function(func_expression: str):
    """
    Parses function expression
    :param func_expression: expression with syntax 'def func_name(arg1, arg2, ..., argN): return ...
    :return: parsed expression
    """

    no_def = func_expression[4:]
    args: list[tuple[str, str | None]] = []
    got_default_values = False
    name = ""
    cur_argument_name = ""
    cur_argument_default = None
    return_word = ""
    min_args = max_args = 0
    for i in range(len(no_def)):
        s = no_def[i]
        if "(" not in name:
            name+=s

        elif ")" not in name:
            if s == ")":
                name+=")"
                args.append((cur_argument_name, cur_argument_default))
                continue
            elif s == ",":
                if cur_argument_default=='':
                    raise SyntaxError(f"{name}: cannot default argument value of {cur_argument_name} empty")
                elif cur_argument_name is None and got_default_values:
                    raise ValueError(f"Non-default argument {cur_argument_name} after default ones")
                args.append((cur_argument_name, cur_argument_default))
                cur_argument_name = ''
                cur_argument_default = None
                continue

            elif s == "=":
                cur_argument_default = ''
                got_default_values = True
                min_args-=1
                continue
            if s == " ":
                continue
            if cur_argument_default is not None:
                cur_argument_default += s
            else:

                if cur_argument_name == '':
                    min_args += 1 * int(not(got_default_values))
                    max_args+=1
                cur_argument_name += s

        else:
            if s == " " and return_word != ":return":
                continue
            elif s == " " and return_word == ":return":
                exp = no_def[i:].strip()
                name = name[:-2]
                return name, args, exp, min_args, max_args
            else:
                return_word += s
                if return_word not in ":return":
                    raise InvalidTokenError(f"Invalid token when defining return value of function {name}: {return_word}", exc_type="forbidden_symbol")


class CompiledExpression(CallAllMethods):
    def __init__(self, expression: str, var_map = None, func_map = None):
        self.expression = expression
        self.var_map = var_map or {}
        self.func_map = func_map or {}
        self.expression = self.expression.replace(",)", ")")
        self.call_all_methods()
        self.expression = self.expression.replace(" ", "")

    def _compile_vars_and_functions(self):
        splitted = self.expression.split(";")
        if len(splitted)==1:
            return
        for var in splitted[:-1]:
            var = var.lstrip()
            if var.count(" let ")+('; '+var).count(";let ")+var.count(" def ")+(';'+var).count("def")>0:
                raise SyntaxError(f"Unseparated variable(or function) and variable(or function) defines: {var}")


            if var.startswith("let"):

                var_name, var_val = var[4:].split("=")
                var_name = var_name.replace(" ", "")
                var_val = var_val.replace(" ", "")
                self.var_map[var_name] = var_val
                if var_name in self.func_map.keys():
                    raise VariableOvershadowError(f"Variable '{var_name}' overshadows function '{var_name}'")
                elif var_name.isdigit():
                    raise VariableOvershadowError(f"Cannot name variables with numbers: '{var_name} = {var_val}'")
            elif var.startswith("def"):
                name, args, expr, min_args, max_args = parse_function(var)
                if name in self.var_map.keys():
                    raise VariableOvershadowError(f"Variable '{name}' overshadows function '{name}'")
                elif name.isdigit():
                    raise VariableOvershadowError(f"Cannot name functions with numbers: 'def {name}'()")

                self.func_map[name] = (args, expr, min_args, max_args)


        self.expression = splitted[-1]
