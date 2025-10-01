from extra.utils import CallAllMethods

class CompiledExpression(CallAllMethods):
    def __init__(self, expression: str, var_map = None):
        self.expression = expression
        self.var_map = var_map or {}
        self.call_all_methods()
        self.expression = self.expression.replace(" ", "")
        self.expression = self.expression.replace(",)", ")")

    def _compile_vars(self):
        splitted = self.expression.split(";")
        if len(splitted)==1:
            return
        var_map: list[list[str]] = []
        for var in splitted[:-1]:
            var_name, var_val = var[4:].split("=")
            var_name = var_name.replace(" ", "")
            var_val = var_val.replace(" ", "")
            var_map.append([var_name, var_val])
        var_map = list(sorted(var_map, key=lambda item: len(item[0]), reverse=True))
        for i in range(len(var_map)):
            k, v = var_map[i]
            self.var_map[k] = v

        self.expression = splitted[-1]
