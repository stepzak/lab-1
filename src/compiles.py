import constants as cst

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
    var_map = list( sorted(var_map, key=lambda item: len(item[0]), reverse = True))
    for i in range(len(var_map)):
        k, v = var_map[i]
        expression = expression.replace(k, v)
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
