from typing import Any, Literal
import vars
from constants import NAME_FORBIDDEN_SYMBOLS, SYSTEM_NAMES
from extra.exceptions import InvalidTokenError, VariableOvershadowError
from extra.types import Variable
from extra.utils import CallAllMethods

def parse_function(func_expression: str) -> tuple[str, list[tuple[str, str | None]], str, int | Any, int] | None:
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
            if s in vars.OPERATORS.keys():
                raise InvalidTokenError(f"Symbol '{s}' is forbidden for name", exc_type="invalid_token")
            if s == ")":
                name+=")"
                if not cur_argument_name:
                    continue
                if cur_argument_default is None and got_default_values:
                    raise ValueError(f"'{func_expression}': Non-default argument '{cur_argument_name}' after default ones")
                args.append((cur_argument_name, cur_argument_default))
                continue
            elif s == ",":
                if cur_argument_default=='':
                    raise SyntaxError(f"{name}: cannot default argument value of '{cur_argument_name}' empty")
                elif cur_argument_default is None and got_default_values:
                    raise ValueError(f"'{func_expression}': Non-default argument '{cur_argument_name}' after default ones")
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
                    min_args += 1 * int(not got_default_values)
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
                    raise InvalidTokenError(f"Invalid token when defining return value of function '{name}': '{return_word}'", exc_type="forbidden_symbol")
    return None


def check_valid_name(name: str, typeof: Literal["operator", "function", "variable", "argument"], var_map: dict, func_map: dict, op_map: dict):
    checks_config = {
        "operator": [(var_map, "variable"), (func_map, "function")],
        "function" : [(var_map, "variable"), (op_map, "operator"),],
        "variable" : [(op_map, "operator"), (func_map, "function")],
        "argument" : [(op_map, "operator"), (func_map, "function")],
    }

    cfg = checks_config[typeof]
    cfg.extend([(vars.FUNCTIONS_CALLABLE_ENUM, "function"), (vars.OPERATORS, "operator")])

    try:
        float(name[0])
        raise VariableOvershadowError(f"{typeof} '{name}' overshadows a number")
    except ValueError:
        pass

    intersection_set = set(name) & NAME_FORBIDDEN_SYMBOLS
    if name in SYSTEM_NAMES:
        raise VariableOvershadowError(f"{typeof} '{name}' is forbidden: it is a system name")
    if intersection_set:
        raise InvalidTokenError(f"{typeof} '{name}': symbols {intersection_set} are forbidden in naming", exc_type="forbidden_symbol")

    for check in cfg:
        if name in check[0].keys():
            raise VariableOvershadowError(f"{typeof} '{name}' overshadows {(typeof==check[1])*'default '}{check[1]} '{name}'")


def parse_operator(operator_expression: str): #operator '->': 1, l+r*4-1,false;  2 -> 3 = 2+3*4-1 = 13
    """
    Converts operator expression to (sign, priority, function, is_right_associative)\n
    e.g. operator expression "operator '->': 1, l<=r, False" \n will be converted to (->, 1, l<=r, False)\n
    where l is left operand, r is right operand
    :param operator_expression: operator expression
    :return (sign, priority, function, is_right_associative)
    """
    shift = len("operator")
    sliced = operator_expression[shift:].strip()
    current_op_name = ""
    par_count = sliced.count("'")
    if par_count!=2:
        raise SyntaxError(f"""Invalid operator expression {sliced}: must have exactly two "'" but {par_count} found""")
    elif sliced.find("'")!=0:
        raise SyntaxError(f"""Invalid syntax for operator expression {sliced}: operator sign must start with "'" """)

    for i in range(len(sliced)):
        s = sliced[i]
        if s in "()":
            raise InvalidTokenError("Operator name cannot contain brackets", exc_type="forbidden_symbol")
        if len(current_op_name)>1:
            if current_op_name[-1]=="'":
                if i==" ":
                    continue
                expr = sliced[i+1:].strip()
                try:
                    priority, full_expr, right_assoc = expr.split(",")
                except ValueError:
                    raise SyntaxError(
                        f"""Invalid syntax for defining operator {current_op_name}: {expr}, template: operator 'operator_sign': (priority), (expression via l and r), (is right associative)""")
                return current_op_name.replace("'", "").strip(), float(priority), full_expr, True if right_assoc.strip() in ["True", "true", "1"] else False

        if s == "'":
            current_op_name+="'"
        elif current_op_name.startswith("'") and s!="'":
            current_op_name += s
            if s==" ":
                raise SyntaxError("Operator sign cannot contain whitespaces")
    return None


class CompiledExpression(CallAllMethods):
    def __init__(self, expression: str, var_map = None, func_map = None, op_map = None):
        self.expression = expression
        self.var_map = var_map or {}
        self.func_map = func_map or {}
        self.op_map = op_map or {}
        self.expression = self.expression.replace(",)", ")")
        self.call_all_methods()

    def _compile_vars_and_functions(self):
        splitted = self.expression.split(";")
        if len(splitted)==1:
            return
        for var in splitted[:-1]:
            var = var.lstrip()
            if var.count(" let ")+var.count(" def ")+var.count(" operator ")>0:
                raise SyntaxError(f"Unseparated variable(or function/operator) and variable(or function/operator) defines: {var}")

            if var.startswith("let"):

                var = var[4:]
                first_eq = var.find("=")
                if first_eq == -1:
                    raise SyntaxError(f"{var}: cannot define variable without '='")
                var_name, var_val = var[:first_eq], var[first_eq+1:]

                var_name = var_name.strip()
                var_val = var_val.strip()
                if len(var_name) == 0:
                    raise SyntaxError(f"{var}: cannot define variable with empty name")
                if len(var_val) == 0:
                    raise SyntaxError(f"{var}: variable cannot be empty")

                for op in vars.OPERATORS:
                    if var_name.find(op) != -1:
                        raise InvalidTokenError(f"Variable name '{var_name}' cannot contain operators('{op}')", exc_type="forbidden_symbol")

                check_valid_name(var_name, "variable", self.var_map, self.func_map, self.op_map)
                self.var_map[var_name] = Variable(var_val, False)

            elif var.startswith("def"):
                name, args, expr, min_args, max_args = parse_function(var)
                check_valid_name(name, "function", self.var_map, self.func_map, self.op_map)
                for arg in args:
                    try:
                        check_valid_name(arg[0], "argument", self.var_map, self.func_map, self.op_map)
                    except VariableOvershadowError as e:
                        raise VariableOvershadowError(f"Error defining with '{var}': "+str(e))
                self.func_map[name] = (args, expr, min_args, max_args)

            elif var.startswith("operator"):
                name, priority, op_expr, right_assoc = parse_operator(var)
                check_valid_name(name, "operator", self.var_map, self.func_map, self.op_map)
                self.op_map[name] = (priority, op_expr, right_assoc)

        self.expression = splitted[-1]

    def _d_check_starts_with_operators(self): #begins with _d to start after _compile_vars_and_functions
        for op in list(self.op_map.keys())+list(vars.OPERATORS.keys()):
            for func in self.func_map.keys():
                if func.find(op) != -1:
                    raise InvalidTokenError(f"Function name '{func}' cannot contain operators('{op}')", exc_type = "forbidden_symbol")
            for var in self.var_map.keys():
                if var.find(op) != -1:
                    raise InvalidTokenError(f"Variable name '{var}' cannot contain operators('{op}')'", exc_type="forbidden_symbol")

if __name__ == "__main__":
    print(parse_operator("operator '=:': 1 l+r:=1"))
