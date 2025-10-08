import decimal
import logging
from sys import stdout
import src.constants as cst
from src.calculator import Calculator
from src.extra.utils import check_is_integer, round_decimal


def main():
    """
    Entry point for application. Gets expression for stdin and passes it to calc() function
    """
    while True:

        expression: str = input("Enter the expression to calculate(q to exit): ")
        if expression == "q":
            exit(0)
        try:
            calc = Calculator()
            result = calc.calc(expression=expression)
        except Exception:
            result = None
        try:
            if type(result) is not int and result:
                result = round_decimal(result)  # type: ignore
                if check_is_integer(result):
                    result = int(result)

            if cst.SCIENTIFIC_FORM and result:
                result = ("{:."+str(cst.SCIENTIFIC_FORM)+"e}").format(decimal.Decimal(result))
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
    main()
