from inspect import signature

def f(k, v):
    return

s = signature(f)

for param in s.parameters.items():
    print(param)
