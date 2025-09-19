import logging
import re
from sys import stdout

import constants as c

logger = logging.getLogger(__name__)
logging.basicConfig(
    format = c.FORMAT,
    stream = stdout
)

def main():
    """Entry point for application"""
    while True:
        expression = input("Enter the expression to calculate: ")

        #parentheses check
        parentheses_total = 0

        for s in expression:
            if s=="(":
                parentheses_total+=1
            elif s==")":
                parentheses_total-=1
            if parentheses_total<0:
                break
        if parentheses_total==0:
            break
        logging.error("Invalid parenthesis syntax")


if __name__ == "__main__":
    basic_expr = "-2*3+(5+6)/7^3"
    main()
