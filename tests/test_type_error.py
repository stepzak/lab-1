from tests.test_ok import suppress_error


def test_two_binary_operations():
    expression = "3*/3"
    suppress_error(expression, TypeError)

def test_invalid_args_max():
    expression = "max()"
    suppress_error(expression, TypeError)

def test_invalid_args_min():
    expression = "min()"
    suppress_error(expression, TypeError)

def test_invalid_args_abs():
    expression = "abs(1, 2)"
    suppress_error(expression, TypeError)

def test_invalid_args_pow_few():
    expression = "pow(2)"
    suppress_error(expression, TypeError)

def test_invalid_args_pow_many():
    expression = "pow(1, 2, 3, 4)"
    suppress_error(expression, TypeError)

def test_invalid_args_pow_float_3():
    expression = "pow(3.1, 3, 2)"
    suppress_error(expression, TypeError)

def test_float_mod_left():
    expression = "7.3%3"
    suppress_error(expression, TypeError)

def test_float_mod_right():
    expression = "7%3.3"
    suppress_error(expression, TypeError)

def test_float_int_div_left():
    expression = "7.3//3"
    suppress_error(expression, TypeError)

def test_float_int_div_right():
    expression = "7//33.3"
    suppress_error(expression, TypeError)

def test_float_arg1_3args():
    expression = "pow(1.1, 2, 2)"
    suppress_error(expression, TypeError)

def test_float_arg2_3args():
    expression = "pow(3, 2.2, 2)"
    suppress_error(expression, TypeError)
