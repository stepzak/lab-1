import decimal
import logging
from sys import stdout

import constants as cst # type: ignore
from calculator import Calculator

def main():
    """
    Entry point for application. Gets expression for stdin and passes it to calc() function
    """
    while True:

        expression: str = input("Enter the expression to calculate(q to exit): ")
        if expression == "q":
            exit(0)
        try:
            calc = Calculator(expression)
            result = calc.calc()
        except Exception:
            result = None
        try:
            if cst.SCIENTIFIC_FORM and result:
                result = ("{:."+str(cst.SCIENTIFIC_FORM)+"E}").format(decimal.Decimal(result))
            logger.info(f"{expression} = {result}")
        except ValueError:
            logger.error(f"Could not calculate expression {expression}: too many digits")
            raise


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(cst.LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(stdout),
        ],
        format=cst.FORMAT
    )
    #logger.info(tokenize(CompiledExpression("max((2//3), 2)")))
    main()
