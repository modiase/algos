"""
Given a string, return whether it represents a number. Here are the different kinds of numbers:

"10", a positive integer
"-10", a negative integer
"10.1", a positive real number
"-10.1", a negative real number
"1e5", a number in scientific notation
And here are examples of non-numbers:

"a"
"x 1"
"a -2"
"-"
"""
import re
import math

integer_regex = re.compile(r'(?P<sign>[-]?)(?P<integral>\d+)')
real_regex = re.compile(r'(?P<sign>[-]?)(?P<integral>\d+)\.(?P<decimal>\d+)')
scientific_regex = re.compile(
    r'(?P<sign>[-]?)(?P<coefficient>\d+(\.\d+)?)e(?P<exponent>\d+)')


def parse_integer(mo: re.Match):
    sign = bool(mo.group('sign')) and -1 or 1
    integral = mo.group('integral')
    return sign*int(integral)


def parse_real(mo: re.Match):
    sign = bool(mo.group('sign')) and -1 or 1
    integral = int(mo.group('integral'))
    decimal = mo.group('decimal')
    decimal = int(decimal) / math.pow(10, len(decimal))

    return sign*(integral + decimal)


def parse_scientific(mo: re.Match):
    sign = bool(mo.group('sign')) and -1 or 1
    coefficient = parse_number(mo.group('coefficient'))
    exponent = int(mo.group('exponent'))
    return sign * coefficient * math.pow(10, exponent)


def parse_number(s):
    if mo := integer_regex.fullmatch(s):
        return parse_integer(mo)
    elif mo := real_regex.fullmatch(s):
        return parse_real(mo)
    elif mo := scientific_regex.fullmatch(s):
        return parse_scientific(mo)
    else:
        return None


assert parse_number('22') == 22
assert parse_number('3.1415') == 3.1415
assert parse_number('1.23e4') == 12300
assert parse_number('a') is None
assert parse_number('x 1') is None
assert parse_number('a -2') is None
assert parse_number('-') is None
assert parse_number('"') is None
